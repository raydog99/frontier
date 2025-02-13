open Torch

type data_matrix = Tensor.t
type covariance = Tensor.t

type config = {
    tol: float;
    max_iter: int;
    min_eigenval: float;
    regularization: float;
}

let default_config = {
    tol = 1e-6;
    max_iter = 1000;
    min_eigenval = 1e-10;
    regularization = 1e-4;
}

type estimation_result = {
    estimate: Tensor.t;
    iterations: int;
    error: float;
    converged: bool;
}

type rank_result = {
    rank: int;
    eigenvalues: Tensor.t;
    explained_variance: float array;
}

let center_data (x: data_matrix) : data_matrix =
    let means = Tensor.mean x ~dim:[0] ~keepdim:true in
    Tensor.(x - means)

let sample_covariance (x: data_matrix) : covariance =
    let n = Tensor.size x 0 in
    let x_centered = center_data x in
    Tensor.(mm (transpose2 x_centered 0 1) x_centered /. float n)

let stable_svd (x: data_matrix) (config: config) =
    let p = Tensor.size x 1 in
    let reg_matrix = Tensor.(eye p * config.regularization) in
    let x_reg = Tensor.(mm (transpose2 x 0 1) x + reg_matrix) in
    try
      let u, s, v = Tensor.svd x_reg ~some:true in
      let s_filtered = Tensor.map s ~f:(fun x -> 
        if x < config.min_eigenval then config.min_eigenval else x) in
      Some (u, s_filtered, v)
    with _ -> None

let rsvp (x: data_matrix) : covariance =
    let x_centered = center_data x in
    let _, _, v = Tensor.svd x_centered ~some:true in
    Tensor.(mm (transpose2 v 0 1) v)

let rsvp_subsample (x: data_matrix) ~(m: int) ~(b: int) : covariance =
    let p = Tensor.size x 1 in
    let sum = List.init b (fun _ ->
      let indices = Tensor.randperm m ~dtype:Int64 in
      let subsample = Tensor.index_select x 0 indices in
      rsvp subsample
    ) |> List.fold_left Tensor.add (Tensor.zeros [p; p]) in
    Tensor.(sum /. float b)

let rsvp_split (x: data_matrix) ~(m: int) : covariance =
    let n = Tensor.size x 0 in
    let p = Tensor.size x 1 in
    let b = n / m in
    let splits = List.init b (fun i ->
      let start_idx = i * m in
      let split = Tensor.narrow x 0 start_idx m in
      rsvp split
    ) in
    let sum = List.fold_left Tensor.add (Tensor.zeros [p; p]) splits in
    Tensor.(sum /. float b)

let clear_prefix_eigenvals (ell: int) (d_squared: Tensor.t) : Tensor.t =
    let result = Tensor.clone d_squared in
    for i = 0 to ell - 1 do
      Tensor.fill_float (Tensor.narrow result 0 i 1) 0.0
    done;
    result

let transform_spectrum (eigenvals: Tensor.t) = function
    | `PCA l -> 
        let result = Tensor.clone eigenvals in
        for i = 0 to l - 1 do
          Tensor.fill_float (Tensor.narrow result 0 i 1) 0.0
        done;
        result
    | `Threshold t ->
        Tensor.map eigenvals ~f:(fun x -> if x < t then 0.0 else x)
    | `Shrinkage alpha ->
        let mean = Tensor.mean eigenvals in
        Tensor.(alpha * mean + ((Tensor.ones_like eigenvals - alpha) * eigenvals))

  let pc_removal (x: Tensor.t) ~(num_components: int) : Tensor.t =
    let n = Tensor.size x 0 in
    let p = Tensor.size x 1 in
    let x_centered = center_data x in
    let u, s, v = Tensor.svd x_centered ~some:true in
    let s_transformed = clear_prefix_eigenvals num_components s in
    let v_t = Tensor.transpose2 v 0 1 in
    Tensor.(mm (mm v (diagonal s_transformed)) v_t /. float n)

(* Nodewise regression *)
let nodewise_regression (sigma_hat: Tensor.t) (j: int) (lambda: float) : Tensor.t =
    let p = Tensor.size sigma_hat 0 in
    let beta = Tensor.zeros [p] in
    
    (* Setup optimization problem *)
    let w = Tensor.get2 sigma_hat j j in
    let w12 = Tensor.select sigma_hat j 0 in
    
    (* Coordinate descent *)
    let max_iter = 1000 in
    let tol = 1e-6 in
    
    for iter = 1 to max_iter do
      let old_beta = Tensor.clone beta in
      
      (* Update each coordinate *)
      for k = 0 to p-1 do
        if k <> j then
          let r = Tensor.(w12 - mm sigma_hat beta) in
          let update = Tensor.(sum (r * Tensor.select sigma_hat k 0)) in
          let soft_thresh x t =
            let sign = if x > 0.0 then 1.0 else -1.0 in
            sign *. max 0.0 (abs_float x -. t)
          in
          Tensor.fill_float (Tensor.narrow beta 0 k 1) 
            (soft_thresh (Tensor.float_value update) lambda)
      done;
      
      if Tensor.(norm (beta - old_beta)) < tol then
        iter = max_iter
    done;
    
    beta

(* Estimate CIG *)
let estimate_cig (x: Tensor.t) (lambda: float) : Tensor.t =
    let p = Tensor.size x 1 in
    let sigma_hat = rsvp x in
    
    let edges = Tensor.zeros [p; p] in
    for j = 0 to p-1 do
      let beta_j = nodewise_regression sigma_hat j lambda in
      
      (* Set edges based on non-zero coefficients *)
      for k = 0 to p-1 do
        if k <> j && 
           abs_float (Tensor.float_value (Tensor.get beta_j k)) > 0.0 then begin
          Tensor.fill_float (Tensor.narrow2 edges j k 1 1) 1.0;
          Tensor.fill_float (Tensor.narrow2 edges k j 1 1) 1.0
        end
      done
    done;
    
    edges

(* Nodewise regression with inference *)
let nodewise_regression_inference (x: Tensor.t) (j: int) (lambda: float) 
    : Tensor.t * Tensor.t =
    let n = Tensor.size x 0 in
    let p = Tensor.size x 1 in
    
    (* Compute beta estimates *)
    let sigma_hat = rsvp x in
    let beta = nodewise_regression sigma_hat j lambda in
    
    (* Compute standard errors *)
    let residuals = Tensor.(x - mm x beta) in
    let sigma_e = Tensor.(mm (transpose2 residuals 0 1) residuals /. float n) in
    let x_j = Tensor.narrow x 1 j 1 in
    let inv_gram = Tensor.(inverse (mm (transpose2 x 1) x)) in
    let std_errors = Tensor.sqrt Tensor.(diagonal (mm (mm inv_gram sigma_e) inv_gram)) in
    
    (beta, std_errors)

  (* Partial correlation computation *)
  let partial_correlation (sigma: Tensor.t) (i: int) (j: int) (s: int list) : float =
    let indices = i :: j :: s in
    let sub_cov = Tensor.index_select2 sigma indices indices in
    let prec = Tensor.inverse sub_cov in
    let rho = -.(Tensor.float_value (Tensor.get2 prec 0 1)) /.
              sqrt (Tensor.float_value (Tensor.get2 prec 0 0) *. 
                   Tensor.float_value (Tensor.get2 prec 1 1)) in
    rho

(* Build conditional graph *)
let build_conditional_graph (sigma: Tensor.t) (alpha: float) : Tensor.t * (int * int * int list) list =
    let p = Tensor.size sigma 0 in
    let adj_matrix = Tensor.ones [p; p] in
    let separating_sets = ref [] in
    
    (* Initialize diagonal to zero *)
    for i = 0 to p-1 do
      Tensor.fill_float (Tensor.narrow2 adj_matrix i i 1 1) 0.0
    done;

    (* Test for conditional independence *)
    let max_size = p - 2 in
    for size = 0 to max_size do
      for i = 0 to p-1 do
        for j = i+1 to p-1 do
          if Tensor.float_value (Tensor.get2 adj_matrix i j) = 1.0 then
            (* Get adjacent nodes *)
            let adj_i = List.init p (fun k -> k)
                       |> List.filter (fun k -> 
                            k <> i && k <> j && 
                            Tensor.float_value (Tensor.get2 adj_matrix i k) = 1.0) in
            
            (* Test all conditioning sets of current size *)
            let cond_sets = combinations size adj_i in
            List.iter (fun s ->
              let rho = partial_correlation sigma i j s in
              if abs_float rho < alpha then begin
                Tensor.fill_float (Tensor.narrow2 adj_matrix i j 1 1) 0.0;
                Tensor.fill_float (Tensor.narrow2 adj_matrix j i 1 1) 0.0;
                separating_sets := (i, j, s) :: !separating_sets
              end
            ) cond_sets
        done
      done
    done;
    
    (adj_matrix, !separating_sets)

(* V-structure identification *)
let identify_v_structures (conditional_graph: Tensor.t) 
                          (sep_sets: (int * int * int list) list) : Tensor.t =
    let p = Tensor.size conditional_graph 0 in
    let cpdag = Tensor.clone conditional_graph in
    
    for i = 0 to p-1 do
      for j = 0 to p-1 do
        for k = 0 to p-1 do
          if i <> j && j <> k && i <> k &&
             Tensor.float_value (Tensor.get2 conditional_graph i j) = 1.0 &&
             Tensor.float_value (Tensor.get2 conditional_graph j k) = 1.0 &&
             Tensor.float_value (Tensor.get2 conditional_graph i k) = 0.0 then
            let is_v_structure = 
              not (List.exists (fun (x, y, s) ->
                (x = i && y = k || x = k && y = i) &&
                List.mem j s
              ) sep_sets) in
            
            if is_v_structure then begin
              Tensor.fill_float (Tensor.narrow2 cpdag i j 1 1) 2.0;
              Tensor.fill_float (Tensor.narrow2 cpdag k j 1 1) 2.0
            end
        done
      done
    done;
    
    cpdag

(* Edge orientation *)
let orient_edges (cpdag: Tensor.t) : Tensor.t =
    let p = Tensor.size cpdag 0 in
    let oriented = Tensor.clone cpdag in
    let changed = ref true in
    
    while !changed do
      changed := false;
      
      (* Orient chain i -> j - k to i -> j -> k *)
      for i = 0 to p-1 do
        for j = 0 to p-1 do
          for k = 0 to p-1 do
            if i <> j && j <> k && i <> k &&
               Tensor.float_value (Tensor.get2 oriented i j) = 2.0 &&
               Tensor.float_value (Tensor.get2 oriented j k) = 1.0 &&
               Tensor.float_value (Tensor.get2 oriented i k) = 0.0 then begin
              Tensor.fill_float (Tensor.narrow2 oriented j k 1 1) 2.0;
              changed := true
            end
          done
        done
      done;
      
      (* Orient common cause i - j -> k to i -> j -> k *)
      for i = 0 to p-1 do
        for j = 0 to p-1 do
          for k = 0 to p-1 do
            if i <> j && j <> k && i <> k &&
               Tensor.float_value (Tensor.get2 oriented i j) = 1.0 &&
               Tensor.float_value (Tensor.get2 oriented j k) = 2.0 &&
               Tensor.float_value (Tensor.get2 oriented i k) = 0.0 then begin
              Tensor.fill_float (Tensor.narrow2 oriented i j 1 1) 2.0;
              changed := true
            end
          done
        done
      done
    done;
    
    oriented

(* Get descendants of a node *)
let get_descendants (graph: Tensor.t) (node: int) : int list =
    let p = Tensor.size graph 0 in
    let descendants = ref [] in
    let visited = Array.make p false in
    
    let rec dfs current =
      visited.(current) <- true;
      for next = 0 to p-1 do
        if not visited.(next) && 
           Tensor.float_value (Tensor.get2 graph current next) = 2.0 then begin
          descendants := next :: !descendants;
          dfs next
        end
      done
    in
    
    dfs node;
    !descendants

(* Path coefficient computation *)
let compute_path_coefficient (graph: Tensor.t) 
                             (sigma: Tensor.t) 
                             (start: int) 
                             (end_: int) : float =
    let paths = find_directed_paths graph start end_ in
    
    List.fold_left (fun acc path ->
      let path_coef = List.fold_left2 (fun c i j ->
        let beta = Tensor.float_value (Tensor.get2 sigma i j) /.
                  Tensor.float_value (Tensor.get2 sigma i i) in
        c *. beta
      ) 1.0 path (List.tl path) in
      acc +. path_coef
    ) 0.0 paths

(* Find directed paths *)
let find_directed_paths (graph: Tensor.t) (start: int) (end_: int) : int list list =
    let p = Tensor.size graph 0 in
    let paths = ref [] in
    
    let rec dfs current path =
      if current = end_ then
        paths := (List.rev (current :: path)) :: !paths
      else begin
        for next = 0 to p-1 do
          if not (List.mem next path) && 
             Tensor.float_value (Tensor.get2 graph current next) = 2.0 then
            dfs next (current :: path)
        done
      end
    in
    
    dfs start [];
    !paths