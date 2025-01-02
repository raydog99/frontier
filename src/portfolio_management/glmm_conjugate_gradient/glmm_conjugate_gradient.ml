open Torch

type precision_matrix = {
  matrix: Tensor.t;
  dim: int;
  sparsity: float;
}

type design_matrix = {
  matrix: Tensor.t;
  n_obs: int;
  n_params: int;
}

type glmm_params = {
  tau: float;
  theta: Tensor.t;
  prior_mean: Tensor.t;
  prior_precision: Tensor.t;
}

type model_config = {
  n_factors: int;
  factor_sizes: int array;
  fixed_effects: int;
}

type observation = {
  response: float;
  factor_levels: int array;
}

let get_block matrix ~row_start ~row_end ~col_start ~col_end =
  let rows = row_end - row_start + 1 in
  let cols = col_end - col_start + 1 in
  let block = Tensor.zeros [rows; cols] in
  for i = 0 to rows - 1 do
    for j = 0 to cols - 1 do
      let value = Tensor.get matrix [|row_start + i; col_start + j|] in
      Tensor.set block [|i; j|] value
    done
  done;
  block

let normalize_adjacency a =
  let d = Tensor.sum a ~dim:[1] ~keepdim:true in
  let d_sqrt_inv = Tensor.pow d (-0.5) in
  let d_mat = Tensor.diag (Tensor.squeeze d_sqrt_inv ~dim:[1]) in
  Tensor.mm (Tensor.mm d_mat a) d_mat

let sparse_multiply a b =
  Tensor.mm a b

let compute_sparsity tensor =
  let total = Tensor.numel tensor in
  let non_zero = Tensor.(sum (tensor != float 0.)) |> Tensor.float_value in
  non_zero /. (float_of_int total)

let factorize q =
  try 
    let l = Tensor.cholesky q.matrix ~upper:false in
    Some { matrix = l; dim = q.dim; sparsity = MatrixOps.compute_sparsity l }
  with _ -> None

let column_wise_cholesky q =
  let n = q.dim in
  let l = Tensor.zeros [n; n] in
  let eps = 1e-12 in
  
  let rec compute_column m =
    if m >= n then Some { matrix = l; dim = n; sparsity = MatrixOps.compute_sparsity l }
    else begin
      try
        (* Compute L_mm *)
        let sum_squares = ref 0. in
        for k = 0 to m - 1 do
          let lmk = Tensor.get l [|m; k|] |> Tensor.float_value in
          sum_squares := !sum_squares +. lmk *. lmk
        done;
        let qmm = Tensor.get q.matrix [|m; m|] |> Tensor.float_value in
        let lmm_sq = qmm -. !sum_squares in
        
        if lmm_sq <= eps then None
        else begin
          let lmm = sqrt lmm_sq in
          Tensor.set l [|m; m|] (Tensor.float lmm);

          (* Compute remaining entries in column m *)
          for j = (m + 1) to n - 1 do
            let sum_products = ref 0. in
            for k = 0 to m - 1 do
              let lmk = Tensor.get l [|m; k|] |> Tensor.float_value in
              let ljk = Tensor.get l [|j; k|] |> Tensor.float_value in
              sum_products := !sum_products +. lmk *. ljk
            done;
            let qjm = Tensor.get q.matrix [|j; m|] |> Tensor.float_value in
            let ljm = (qjm -. !sum_products) /. lmm in
            Tensor.set l [|j; m|] (Tensor.float ljm)
          done;
          compute_column (m + 1)
        end
      with _ -> None
    end
  in
  compute_column 0

let solve ~l ~b =
  let n = Tensor.size l [0] in
  let y = Tensor.zeros [n] in
  
  (* Forward substitution *)
  for i = 0 to n - 1 do
    let sum = ref 0. in
    for j = 0 to i - 1 do
      let lij = Tensor.get l [|i; j|] |> Tensor.float_value in
      let yj = Tensor.get y [|j|] |> Tensor.float_value in
      sum := !sum +. lij *. yj
    done;
    let bi = Tensor.get b [|i|] |> Tensor.float_value in
    let lii = Tensor.get l [|i; i|] |> Tensor.float_value in
    Tensor.set y [|i|] (Tensor.float ((bi -. !sum) /. lii))
  done;

  (* Backward substitution *)
  let x = Tensor.zeros [n] in
  for i = n - 1 downto 0 do
    let sum = ref 0. in
    for j = i + 1 to n - 1 do
      let lji = Tensor.get l [|j; i|] |> Tensor.float_value in
      let xj = Tensor.get x [|j|] |> Tensor.float_value in
      sum := !sum +. lji *. xj
    done;
    let yi = Tensor.get y [|i|] |> Tensor.float_value in
    let lii = Tensor.get l [|i; i|] |> Tensor.float_value in
    Tensor.set x [|i|] (Tensor.float ((yi -. !sum) /. lii))
  done;
  x

let estimate_cost q =
  match factorize q with
  | None -> None
  | Some l ->
      let nl = float_of_int (Tensor.(sum (l.matrix != float 0.)) 
                           |> Tensor.float_value 
                           |> int_of_float) in
      let p = float_of_int q.dim in
      Some (nl *. nl /. p, nl ** 1.5)

module ConjugateGradient = struct
  type convergence_info = {
    iterations: int;
    errors: float array;
    bound_ratio: float;
  }

  let solve ~q ~b ?(max_iter=1000) ?(tol=1e-6) =
    let x = Tensor.zeros [q.dim] in
    let r = Tensor.sub b (Tensor.mm q.matrix x) in
    let p = r in
    let errors = Array.make (max_iter + 1) 0. in
    let initial_error = Tensor.norm r |> Tensor.float_value in
    errors.(0) <- initial_error;

    let rec iterate x r p iter =
      if iter >= max_iter then 
        x, {iterations=iter; errors; bound_ratio=0.}
      else begin
        let ap = Tensor.mm q.matrix p in
        let alpha = Tensor.(dot r r /. dot p ap) in
        let x_next = Tensor.(add x (mul_scalar p alpha)) in
        let r_next = Tensor.(sub r (mul_scalar ap alpha)) in
        let err = Tensor.norm r_next |> Tensor.float_value in
        errors.(iter + 1) <- err;

        if err < tol *. initial_error then begin
          let condition_number = 
            let eigenvals = Tensor.symeig q.matrix ~eigenvectors:false in
            let max_eig = Tensor.max eigenvals |> Tensor.float_value in
            let min_eig = Tensor.min eigenvals |> Tensor.float_value in
            max_eig /. min_eig in
          let theoretical_bound = 2. *. 
            ((sqrt condition_number -. 1.) /. 
             (sqrt condition_number +. 1.)) ** float_of_int iter in
          x_next, {
            iterations = iter + 1;
            errors = Array.sub errors 0 (iter + 2);
            bound_ratio = err /. (initial_error *. theoretical_bound)
          }
        end else begin
          let beta = Tensor.(dot r_next r_next /. dot r r) in
          let p_next = Tensor.(add r_next (mul_scalar p beta)) in
          iterate x_next r_next p_next (iter + 1)
        end
      end
    in
    iterate x r p 0

  let sample ~q ~mean =
    let z = Tensor.randn [q.dim] in
    let perturbed_b = Tensor.add mean (Tensor.mm q.matrix z) in
    let solution, _ = solve ~q ~b:perturbed_b in
    solution
end

module RandomIntercept = struct
  type model = {
    config: Types.model_config;
    observations: Types.observation array;
    design: Types.design_matrix;
    params: Types.glmm_params;
    tau_response: float;
    prior_taus: float array;
  }

  let create_z_matrix config observations =
    let n = Array.length observations in
    let total_levels = Array.fold_left (+) 0 config.factor_sizes in
    let z = Tensor.zeros [n; total_levels] in
    
    let offset = ref 0 in
    for k = 0 to config.n_factors - 1 do
      for i = 0 to n - 1 do
        let level = observations.(i).factor_levels.(k) in
        Tensor.set z [|i; !offset + level|] (Tensor.float 1.);
      done;
      offset := !offset + config.factor_sizes.(k)
    done;
    z

  let create config n_obs tau =
    let observations = Array.init n_obs (fun _ ->
      {
        response = 0.0;  (* Initialized to zero *)
        factor_levels = Array.init config.n_factors (fun k ->
          Random.int config.factor_sizes.(k))
      }) in
    
    let z = create_z_matrix config observations in
    let ones = Tensor.ones [n_obs; config.fixed_effects] in
    let design_matrix = Tensor.cat [ones; z] ~dim:1 in
    
    let total_params = Array.fold_left (+) config.fixed_effects config.factor_sizes in
    {
      config;
      observations;
      design = {
        matrix = design_matrix;
        n_obs;
        n_params = total_params;
      };
      params = {
        tau;
        theta = Tensor.zeros [total_params];
        prior_mean = Tensor.zeros [total_params];
        prior_precision = Tensor.eye total_params ~dtype:Float;
      };
      tau_response = 1.0;
      prior_taus = Array.make config.n_factors 1.0;
    }

  let compute_posterior_precision model =
    let vt = Tensor.transpose model.design.matrix ~dim0:0 ~dim1:1 in
    let v_omega = Tensor.mul_scalar model.design.matrix model.tau_response in
    let q = Tensor.add model.params.prior_precision 
            (Tensor.mm vt v_omega) in
    {
      matrix = q;
      dim = model.design.n_params;
      sparsity = MatrixOps.compute_sparsity q;
    }

  let sample_posterior model observations =
    let y = Tensor.of_float1 (Array.map (fun obs -> obs.response) observations) in
    let q = compute_posterior_precision model in
    let mean = Tensor.mm (Tensor.transpose model.design.matrix ~dim0:0 ~dim1:1) 
              (Tensor.mul_scalar y model.tau_response) in
    ConjugateGradient.sample ~q ~mean

  let update_hyperparameters model observations =
    let n = Array.length observations in
    let new_taus = Array.make model.config.n_factors 0.0 in
    let offset = ref model.config.fixed_effects in
    
    (* Update factor precisions *)
    for k = 0 to model.config.n_factors - 1 do
      let size = model.config.factor_sizes.(k) in
      let sum_sq = ref 0. in
      for j = 0 to size - 1 do
        let effect = Tensor.get model.params.theta [|!offset + j|] 
                    |> Tensor.float_value in
        sum_sq := !sum_sq +. effect *. effect
      done;
      let shape = float_of_int size /. 2.0 in
      let rate = !sum_sq /. 2.0 in
      new_taus.(k) <- shape /. rate;
      offset := !offset + size
    done;

    (* Update response precision *)
    let sum_sq_resid = ref 0. in
    let pred = Tensor.mm model.design.matrix model.params.theta in
    Array.iteri (fun i obs ->
      let resid = obs.response -. (Tensor.get pred [|i|] |> Tensor.float_value) in
      sum_sq_resid := !sum_sq_resid +. resid *. resid
    ) observations;
    
    let new_tau_response = float_of_int n /. (2.0 *. !sum_sq_resid) in
    
    new_taus, new_tau_response
end

module SpectralAnalysis = struct
  type eigenvalue_distribution = {
    bulk: float array;
    outliers: float array;
    condition_number: float;
  }

  type spectral_gap = {
    value: float;
    location: int * int;
    relative_size: float;
  }

  let compute_eigenvalues q =
    Tensor.symeig q.matrix ~eigenvectors:false

  let analyze_spectrum q config =
    let eigenvals = compute_eigenvalues q |> Tensor.to_float1 in
    Array.sort compare eigenvals;
    let n = Array.length eigenvals in
    
    (* Separate outliers *)
    let k = config.n_factors in
    let outliers = Array.init (k + 1) (fun i ->
      if i < k then eigenvals.(i)
      else eigenvals.(n - 1)
    ) in
    
    let bulk = Array.init (n - k - 1) (fun i ->
      eigenvals.(i + k)
    ) in

    {
      bulk;
      outliers;
      condition_number = eigenvals.(n-1) /. eigenvals.(0)
    }

  let effective_condition_number distribution ~s ~r =
    let n = Array.length distribution.bulk in
    if s >= n - r then 
      failwith "Invalid s,r parameters for effective condition number";
    distribution.bulk.(n - r - 1) /. distribution.bulk.(s)

  let find_spectral_gaps eigenvalues min_gap_ratio =
    let n = Array.length eigenvalues in
    let gaps = ref [] in
    let avg_spacing = 
      (eigenvalues.(n-1) -. eigenvalues.(0)) /. float_of_int (n-1) in
    
    for i = 0 to n - 2 do
      let gap = eigenvalues.(i+1) -. eigenvalues.(i) in
      let relative = gap /. avg_spacing in
      if relative > min_gap_ratio then
        gaps := {
          value = gap;
          location = (i, i+1);
          relative_size = relative
        } :: !gaps
    done;
    List.rev !gaps
end

module GraphAnalysis = struct
  type graph = {
    vertices: int;
    edges: (int * int) list;
    adjacency: Tensor.t;
  }

  let create_from_precision q =
    let edges = ref [] in
    let n = q.dim in
    let adj = Tensor.zeros [n; n] in
    
    for i = 0 to n - 1 do
      for j = i + 1 to n - 1 do
        if abs_float (Tensor.get q.matrix [|i; j|] |> Tensor.float_value) > 1e-10 then begin
          edges := (i, j) :: !edges;
          Tensor.set adj [|i; j|] (Tensor.float 1.);
          Tensor.set adj [|j; i|] (Tensor.float 1.);
        end
      done
    done;

    { vertices = n; edges = List.rev !edges; adjacency = adj }

  let is_conditionally_independent graph i j sep_set =
    let rec has_path visited v =
      if v = j then true
      else if List.mem v sep_set then false
      else if List.mem v visited then false
      else
        let neighbors = ref [] in
        for k = 0 to graph.vertices - 1 do
          if Tensor.get graph.adjacency [|v; k|] |> Tensor.float_value > 0. then
            neighbors := k :: !neighbors
        done;
        List.exists (has_path (v :: visited)) !neighbors
    in
    not (has_path [i] i)

  let find_potential_fillins graph ordering =
    let n = graph.vertices in
    let fillins = ref [] in
    
    for k = 0 to n - 2 do
      let future = Array.sub ordering (k + 1) (n - k - 1) in
      for i = 0 to Array.length future - 1 do
        for j = i + 1 to Array.length future - 1 do
          let vi = future.(i) in
          let vj = future.(j) in
          if not (List.mem (vi, vj) graph.edges) &&
             not (is_conditionally_independent graph vi vj 
                  (Array.to_list (Array.sub future (j+1) (Array.length future - j - 1))))
          then
            fillins := (vi, vj) :: !fillins
        done
      done
    done;
    !fillins

  let analyze_connectivity graph config =
    let components = Array.make config.n_factors 0 in
    let max_degree = ref 0 in
    
    (* Compute connected components for each factor *)
    let offset = ref config.fixed_effects in
    for k = 0 to config.n_factors - 1 do
      let size = config.factor_sizes.(k) in
      let visited = Array.make size false in
      let component_count = ref 0 in
      
      for i = 0 to size - 1 do
        if not visited.(i) then begin
          incr component_count;
          let queue = Queue.create () in
          Queue.add i queue;
          visited.(i) <- true;
          
          while not (Queue.is_empty queue) do
            let v = Queue.take queue in
            let degree = ref 0 in
            for j = 0 to size - 1 do
              if not visited.(j) && 
                 Tensor.get graph.adjacency [|!offset + v; !offset + j|] 
                 |> Tensor.float_value > 0. 
              then begin
                Queue.add j queue;
                visited.(j) <- true;
                incr degree
              end
            done;
            max_degree := max !max_degree !degree
          done
        end
      done;
      
      components.(k) <- !component_count;
      offset := !offset + size
    done;

    components, float_of_int !max_degree

  let reduce_bandwidth graph =
    let n = graph.vertices in
    let degrees = Array.make n 0 in
    
    (* Compute vertex degrees *)
    List.iter (fun (i, j) ->
      degrees.(i) <- degrees.(i) + 1;
      degrees.(j) <- degrees.(j) + 1;
    ) graph.edges;
    
    (* Cuthill-McKee ordering *)
    let visited = Array.make n false in
    let ordering = Array.make n 0 in
    let idx = ref 0 in
    
    let add_to_ordering v =
      ordering.(!idx) <- v;
      incr idx;
      visited.(v) <- true
    in
    
    (* Find start vertex (minimum degree) *)
    let start = ref 0 in
    let min_deg = ref max_int in
    for i = 0 to n - 1 do
      if degrees.(i) < !min_deg then begin
        start := i;
        min_deg := degrees.(i)
      end
    done;
    
    add_to_ordering !start;
    
    while !idx < n do
      let current = ref (!idx - 1) in
      while !current >= 0 && !idx < n do
        let v = ordering.(!current) in
        (* Get unvisited neighbors sorted by degree *)
        let neighbors = ref [] in
        List.iter (fun (i, j) ->
          let u = if i = v then j else if j = v then i else -1 in
          if u >= 0 && not visited.(u) then
            neighbors := (degrees.(u), u) :: !neighbors
        ) graph.edges;
        List.sort compare !neighbors
        |> List.iter (fun (_, u) -> 
           if not visited.(u) then add_to_ordering u);
        decr current
      done;
      
      (* Handle disconnected components *)
      if !idx < n then begin
        let next = ref 0 in
        while !next < n && visited.(!next) do incr next done;
        if !next < n then add_to_ordering !next
      end
    done;
    
    ordering
end