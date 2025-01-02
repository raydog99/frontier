open Torch

let pi () = Tensor.float_pi

let l2_norm tensor =
  Tensor.pow_tensor_scalar (Tensor.sum (Tensor.pow_tensor_scalar tensor 2.)) 0.5
  
let gradient f x h =
  let dim = Tensor.size x 0 in
  let grad = Tensor.zeros_like x in
  for i = 0 to dim-1 do
    let x_plus = Tensor.copy x in
    let x_minus = Tensor.copy x in
    Tensor.set x_plus [|i|] (Tensor.get x [|i|] +. h);
    Tensor.set x_minus [|i|] (Tensor.get x [|i|] -. h);
    let df = Tensor.sub (f x_plus) (f x_minus) in
    Tensor.set grad [|i|] (Tensor.div_scalar df (2. *. h))
  done;
  grad

let laplacian f x h =
  let dim = Tensor.size x 0 in
  let lap = ref 0. in
  for i = 0 to dim-1 do
    let x_plus = Tensor.copy x in
    let x_minus = Tensor.copy x in
    Tensor.set x_plus [|i|] (Tensor.get x [|i|] +. h);
    Tensor.set x_minus [|i|] (Tensor.get x [|i|] -. h);
    let f_plus = Tensor.to_float0_exn (f x_plus) in
    let f_minus = Tensor.to_float0_exn (f x_minus) in
    let f_center = Tensor.to_float0_exn (f x) in
    lap := !lap +. (f_plus -. 2. *. f_center +. f_minus) /. (h *. h)
  done;
  !lap

module Domain = struct
  type t = {
    dim: int;
    size: float;
    points: Tensor.t array;
    grid_size: int;
  }
  
  let create ~dim ~size =
    let grid_size = max 32 (int_of_float (size *. 40.)) in
    let points = Array.init dim (fun _ ->
      Tensor.linspace (-size/.2.) (size/.2.) grid_size) in
    { dim; size; points; grid_size }
    
  let get_subdomain domain l =
    if l >= domain.size then
      failwith "Subdomain size must be less than domain size";
    let scale = l /. domain.size in
    let grid_size = int_of_float (float_of_int domain.grid_size *. scale) in
    let points = Array.map (fun p ->
      Tensor.narrow p ~dim:0 
        ~start:(domain.grid_size/4) 
        ~length:(domain.grid_size/2)
    ) domain.points in
    { domain with size = l; points; grid_size }
    
  let get_boundary_layer domain =
    let inner = get_subdomain domain (domain.size *. 0.75) in
    let points = Array.map2 (fun o i -> Tensor.sub o i) 
                  domain.points inner.points in
    { domain with points }
    
  let to_indices domain points =
    let scale = float_of_int domain.grid_size /. domain.size in
    Tensor.mul_scalar points scale
    
  let size t = t.size
  let dim t = t.dim
  let grid_size t = t.grid_size
  let points t = t.points
end

module Coefficient = struct
  type t = {
    func: Torch.Tensor.t -> Torch.Tensor.t;
    dim: int;
    period: float array option;
    is_periodic: bool;
  }
  
  let create f = {
    func = f;
    dim = 2;
    period = None;
    is_periodic = false
  }
  
  let create_periodic ~base_func ~period ~dim = {
    func = (fun x ->
      let scaled = Tensor.map (fun pt ->
        let pt_list = Tensor.to_float1_exn pt in
        List.mapi (fun i x ->
          x -. (floor (x /. period.(i))) *. period.(i)
        ) pt_list |> Tensor.of_float1
      ) x ~dim:0 in
      base_func scaled);
    dim;
    period = Some period;
    is_periodic = true
  }
  
  let create_quasiperiodic ~scale =
    let frequencies = [|2. *. pi *. sqrt 2.; 2. *. pi|] in
    let base_func x =
      let dims = Array.to_list (Array.init 2 (fun i ->
        let xi = Tensor.select x 0 i in
        let terms = Array.map (fun freq ->
          Tensor.sin (Tensor.mul_scalar xi freq)
        ) frequencies in
        Array.fold_left Tensor.add (Tensor.zeros_like xi) terms
      )) in
      let sum = List.fold_left Tensor.add 
        (List.hd dims) (List.tl dims) in
      Tensor.mul_scalar (Tensor.exp sum) scale
    in
    create base_func
    
  let evaluate coeff points = coeff.func points
  
  let check_bounds coeff ~alpha ~beta =
    let n_samples = 1000 in
    let points = Tensor.rand [|n_samples; coeff.dim|] in
    let values = evaluate coeff points in
    let min_val = Tensor.min values |> Tensor.to_float0_exn in
    let max_val = Tensor.max values |> Tensor.to_float0_exn in
    alpha <= min_val && max_val <= beta
end

module Regularization = struct
  type t = {
    time_param: float;
    exp_order: int;
    boundary_param: float;
  }
  
  let create ~time_param ~exp_order ~boundary_param =
    { time_param; exp_order; boundary_param }
    
  let optimal_params domain freq =
    let size = Domain.size domain in
    {
      time_param = 1. /. (2. *. pi *. freq);
      exp_order = int_of_float (sqrt (size /. freq));
      boundary_param = exp (-. size *. freq)
    }
end

module Solver = struct
  type t = {
    domain: Domain.t;
    coeff: Coefficient.t;
    regularization: Regularization.t option;
  }
  
  let create domain coeff = {
    domain;
    coeff;
    regularization = None
  }
  
  let with_regularization t reg =
    { t with regularization = Some reg }
    
    let solve_standard t g =
    let n = t.domain.grid_size in
    let h = t.domain.size /. float_of_int n in
    let a = Tensor.zeros [|n; n|] in
    
    (* Build system matrix *)
    for i = 0 to n-1 do
      for j = 0 to n-1 do
        let xi = t.domain.points.(0).(i) in
        let xj = t.domain.points.(0).(j) in
        if i = j then
          let coeff = Coefficient.evaluate t.coeff (Tensor.of_float1 [|xi|]) in
          Tensor.set a [|i; j|] (-2. *. Tensor.to_float0_exn coeff /. (h *. h))
        else if abs(i - j) = 1 then
          let coeff = Coefficient.evaluate t.coeff 
            (Tensor.of_float1 [|min xi xj|]) in
          Tensor.set a [|i; j|] (Tensor.to_float0_exn coeff /. (h *. h))
      done
    done;
    
    MatrixOperations.solve_linear_system 
      { size = n;
        apply = fun x -> Tensor.mv a x }
      g
      
  let solve_regularized t g =
    match t.regularization with
    | None -> solve_standard t g
    | Some reg ->
        let n = t.domain.grid_size in
        let h = t.domain.size /. float_of_int n in
        let a = Tensor.zeros [|n; n|] in
        
        (* Build regularized system *)
        for i = 0 to n-1 do
          for j = 0 to n-1 do
            let xi = t.domain.points.(0).(i) in
            let xj = t.domain.points.(0).(j) in
            let base_val = if i = j then -2. else if abs(i - j) = 1 then 1. else 0. in
            let coeff = Coefficient.evaluate t.coeff (Tensor.of_float1 [|xi|]) in
            Tensor.set a [|i; j|] (base_val *. Tensor.to_float0_exn coeff /. (h *. h))
          done
        done;
        
        let op = { MatrixOperations.size = n;
                  apply = fun x -> Tensor.mv a x } in
        let exp_term = MatrixOperations.apply_matrix_exponential op g reg.time_param in
        let rhs = Tensor.sub g exp_term in
        
        MatrixOperations.solve_linear_system op rhs
        
  let solve_spectral t g =
    let ft = Tensor.fft g in
    let n = t.domain.grid_size in
    let result = Tensor.zeros_like ft in
    
    for k = 0 to n-1 do
      let freq = if k <= n/2 then float_of_int k
                else float_of_int (k - n) in
      let freq_scaled = 2. *. pi *. freq /. t.domain.size in
      
      match t.regularization with
      | None ->
          let op_val = -.(freq_scaled ** 2.) in
          Tensor.set result [|k|] (Tensor.get ft [|k|] /. op_val)
      | Some reg ->
          let op_val = -.(freq_scaled ** 2.) in
          let exp_val = exp (-. op_val *. reg.time_param) in
          Tensor.set result [|k|] (Tensor.get ft [|k|] *. (1. -. exp_val) /. op_val)
    done;
    
    Tensor.ifft result
    
  let compute_l2_error t domain u1 u2 =
    let diff = Tensor.sub u1 u2 in
    let squared_diff = Tensor.pow_tensor_scalar diff 2. in
    let integral = Tensor.sum squared_diff in
    let vol = domain.Domain.size ** float_of_int(t.domain.dim) in
    sqrt (Tensor.to_float0_exn integral *. vol)
end

module ErrorAnalysis = struct
  type error_components = {
    boundary_error: float;
    modeling_error: float;
    total_error: float;
  }
  
  let analyze_error solver u =
    let domain = solver.Solver.domain in
    let interior = Domain.get_subdomain domain (domain.size *. 0.75) in
    
    let boundary_layer = Domain.get_boundary_layer domain in
    let boundary_norm = l2_norm (
      Tensor.narrow u ~dim:0 
        ~start:(domain.grid_size/4) 
        ~length:(domain.grid_size/2)
    ) in
    
    let modeling_err = match solver.regularization with
    | None -> 0.
    | Some reg ->
        let exp_term = Tensor.exp (Tensor.mul_scalar u (-. reg.time_param)) in
        l2_norm exp_term
    in
    
    { boundary_error = Tensor.to_float0_exn boundary_norm;
      modeling_error = modeling_err;
      total_error = sqrt (boundary_norm ** 2. +. modeling_err ** 2.) }
      
  let estimate_convergence_rate sizes errors =
    let n = Array.length sizes - 1 in
    let rates = Array.make n 0. in
    for i = 0 to n-1 do
      rates.(i) <- log (errors.(i) /. errors.(i+1)) /. 
                   log (sizes.(i) /. sizes.(i+1))
    done;
    Array.fold_left (+.) 0. rates /. float_of_int n
    
  let verify_exponential_decay solver u =
    let ft = Tensor.fft u in
    let spec = Tensor.abs ft in
    let decay_rate = SpectralAnalysis.estimate_decay_rate
      { frequencies = [||]; amplitudes = Tensor.to_float1_exn spec; 
        decay_rate = None } in
    decay_rate > 1.0
end

module SpectralAnalysis = struct
  type spectrum = {
    frequencies: float array;
    amplitudes: float array;
    decay_rate: float option;
  }
  
  let compute_spectrum u =
    let ft = Tensor.fft u in
    let spec = Tensor.abs ft in
    let n = Tensor.size spec 0 in
    let freqs = Array.init n (fun i ->
      if i <= n/2 then float_of_int i
      else float_of_int (i - n)
    ) in
    { frequencies = freqs;
      amplitudes = Tensor.to_float1_exn spec;
      decay_rate = None }
      
  let analyze_periodicity coeff domain =
    let points = domain.Domain.points.(0) in
    let values = Coefficient.evaluate coeff points in
    let spec = compute_spectrum values in
    
    let peaks = ref [] in
    let n = Array.length spec.amplitudes in
    for i = 1 to n-2 do
      if spec.amplitudes.(i) > spec.amplitudes.(i-1) &&
         spec.amplitudes.(i) > spec.amplitudes.(i+1) then
        peaksopen Torch

  let estimate_decay_rate spec =
    let n = Array.length spec.amplitudes in
    let high_freq_idx = n / 4 in
    
    (* Fit exponential decay to high frequencies *)
    let x = Array.sub spec.frequencies high_freq_idx (n - high_freq_idx) in
    let y = Array.map log (Array.sub spec.amplitudes high_freq_idx (n - high_freq_idx)) in
    
    let n_points = Array.length x in
    let sum_x = Array.fold_left (+.) 0. x in
    let sum_y = Array.fold_left (+.) 0. y in
    let sum_xy = Array.fold_left2 (fun acc xi yi -> acc +. xi *. yi) 0. x y in
    let sum_xx = Array.fold_left (fun acc xi -> acc +. xi *. xi) 0. x in
    
    let slope = (float_of_int n_points *. sum_xy -. sum_x *. sum_y) /.
               (float_of_int n_points *. sum_xx -. sum_x *. sum_x) in
    -.slope  
end

module Discretization = struct
  type grid = {
    points: Tensor.t array;
    weights: Tensor.t;
  }
  
  let create_grid domain =
    let n = domain.Domain.grid_size in
    let h = domain.Domain.size /. float_of_int n in
    
    (* Compute trapezoidal quadrature weights *)
    let weights = Tensor.ones [|n|] in
    Tensor.set weights [|0|] 0.5;
    Tensor.set weights [|n-1|] 0.5;
    Tensor.mul_scalar_ weights h;
    
    { points = domain.Domain.points; weights }
  
  let interpolate grid points values new_points =
    let n = Tensor.size new_points 0 in
    let result = Tensor.zeros [|n|] in
    
    for i = 0 to n-1 do
      let x = Tensor.get new_points [|i|] in
      (* Find interval containing x *)
      let j = ref 0 in
      while !j < Tensor.size points 0 - 1 && 
            Tensor.get points [|!j + 1|] < x do
        incr j
      done;
      
      let x0 = Tensor.get points [|!j|] in
      let x1 = Tensor.get points [|!j + 1|] in
      let y0 = Tensor.get values [|!j|] in
      let y1 = Tensor.get values [|!j + 1|] in
      
      (* Linear interpolation *)
      let t = (x -. x0) /. (x1 -. x0) in
      Tensor.set result [|i|] (y0 +. t *. (y1 -. y0))
    done;
    
    result
end

module MatrixOperations = struct
  type operator = {
    size: int;
    apply: Tensor.t -> Tensor.t;
  }
  
  let krylov_basis op v m =
    let n = op.size in
    let basis = Array.make m (Tensor.zeros [|n|]) in
    let h = Tensor.zeros [|m; m|] in
    
    (* Normalize first basis vector *)
    basis.(0) <- Tensor.div_scalar v (l2_norm v);
    
    (* Build Krylov subspace *)
    for j = 0 to m-1 do
      let w = op.apply basis.(j) in
      
      (* Orthogonalize against previous vectors (Modified Gram-Schmidt) *)
      for i = 0 to j do
        let hij = Tensor.dot basis.(i) w in
        Tensor.set h [|i; j|] hij;
        let w_new = Tensor.sub w (Tensor.mul_scalar basis.(i) hij) in
        w <- w_new
      done;
      
      if j < m-1 then begin
        let hjp1j = l2_norm w in
        Tensor.set h [|j+1; j|] hjp1j;
        if abs_float hjp1j > 1e-14 then
          basis.(j+1) <- Tensor.div_scalar w hjp1j
      end
    done;
    
    (basis, h)
  
  let apply_matrix_exponential op v t =
    let n = op.size in
    let m = min n 30 in  (* Krylov subspace size *)
    let (basis, h) = krylov_basis op v m in
    
    (* Scale matrix by time parameter *)
    let h_scaled = Tensor.mul_scalar h t in
    
    (* Compute matrix exponential using Padé approximation *)
    let p = 6 in (* Padé order *)
    let denom = ref 1 in
    let exp_h = ref (Tensor.eye m) in
    let term = ref (Tensor.eye m) in
    
    for i = 1 to p do
      term := Tensor.mm !term h_scaled;
      denom := !denom * i;
      let new_term = Tensor.div_scalar !term (float_of_int !denom) in
      exp_h := Tensor.add !exp_h new_term
    done;
    
    (* Compute V * exp(tH) * e1 *)
    let e1 = Tensor.zeros [|m|] in
    Tensor.set e1 [|0|] 1.0;
    let y = Tensor.mv !exp_h e1 in
    
    let result = Tensor.zeros [|n|] in
    for i = 0 to m-1 do
      let yi = Tensor.get y [|i|] in
      Tensor.add_(result, Tensor.mul_scalar basis.(i) yi)
    done;
    
    result
  
  let solve_linear_system op b =
    let n = op.size in
    let tol = 1e-10 in
    let max_iter = min n 100 in
    
    let x = Tensor.zeros [|n|] in
    let r = Tensor.sub b (op.apply x) in
    let beta = l2_norm r in
    
    let rec gmres_iter x r beta m =
      if beta < tol || m >= max_iter then x
      else begin
        let k = min 20 (max_iter - m) in
        let (basis, h) = krylov_basis op r k in
        
        (* Solve least squares problem *)
        let e1 = Tensor.zeros [|k|] in
        Tensor.set e1 [|0|] beta;
        
        let y = Tensor.solve h e1 in
        
        (* Update solution *)
        let update = Tensor.zeros [|n|] in
        for i = 0 to k-1 do
          let yi = Tensor.get y [|i|] in
          Tensor.add_(update, Tensor.mul_scalar basis.(i) yi)
        done;
        
        let x_new = Tensor.add x update in
        let r_new = Tensor.sub b (op.apply x_new) in
        let beta_new = l2_norm r_new in
        
        gmres_iter x_new r_new beta_new (m + k)
      end
    in
    
    gmres_iter x r beta 0
end