open Torch

module type Process = sig
  type t
  val create : unit -> t
  val step : t -> Tensor.t -> Tensor.t
  val get_state : t -> Tensor.t
  val set_state : t -> Tensor.t -> unit
end

(* Forward SDE with direct drift and diffusion *)
module ForwardSDE = struct
  type t = {
    mutable state : Tensor.t;
    time_step : float;
    mutable current_time : float;
    drift : Tensor.t -> float -> Tensor.t;
    diffusion : Tensor.t -> Tensor.t;
  }

  let create initial_state dt drift_fn diff_fn = {
    state = initial_state;
    time_step = dt;
    current_time = 0.0;
    drift = drift_fn;
    diffusion = diff_fn;
  }

  let get_state t = t.state
  let set_state t new_state = t.state <- new_state
  let get_time t = t.current_time

  let step t noise =
    let drift = t.drift t.state t.current_time in
    let diffusion = t.diffusion t.state in
    let increment = 
      (drift * float t.time_step + 
             diffusion * noise * (float t.time_step |> sqrt))
    in
    t.state <- (t.state + increment);
    t.current_time <- t.current_time +. t.time_step;
    t.state
end

(* Backward SDE *)
module BackwardSDE = struct
  type t = {
    mutable y : Tensor.t;
    mutable z : Tensor.t;
    time_step : float;
    mutable current_time : float;
  }

  let create initial_y initial_z dt = {
    y = initial_y;
    z = initial_z;
    time_step = dt;
    current_time = 0.0;
  }

  let get_y t = t.y
  let get_z t = t.z
  let set_y t new_y = t.y <- new_y
  let set_z t new_z = t.z <- new_z

  let step t driver noise =
    let dt = t.time_step in
    let dy = (
      driver * float dt +
      t.z * noise * (float dt |> sqrt)
    ) in
    t.y <- (t.y - dy);
    t.current_time <- t.current_time +. dt;
    t.y, t.z
end

(* Basic numerical schemes *)
module NumericalSchemes = struct
  (* Euler-Maruyama scheme for forward SDE *)
  let euler_maruyama_step x drift diffusion dt dw =
    let dx = (drift * float dt + diffusion * dw) in
    (x + dx)

  (* Explicit scheme for backward SDE *)
  let backward_euler_step y z driver dt dw =
    let dy = (driver * float dt + z * dw) in
    (y - dy)

  (* Generate Brownian increments *)
  let generate_increments dim steps dt =
    let scale = Float.sqrt dt in
    (randn [steps; dim] * float scale)
end

(* Basis functions *)
module BasisFunctions = struct
  type t = {
    functions: (Tensor.t -> Tensor.t) array;
    gradients: (Tensor.t -> Tensor.t) array;
  }

  let create funcs grads = {
    functions = funcs;
    gradients = grads;
  }

  let evaluate t x k =
    t.functions.(k) x

  let evaluate_gradient t x k =
    t.gradients.(k) x

  (* Radial basis function *)
  let rbf_gaussian center width x =
    let diff = (x - center) in
    (exp (-(diff * diff) / (float (2.0 *. width *. width))))

  let rbf_gaussian_gradient center width x =
    let diff = (x - center) in
    let exp_term = (exp (-(diff * diff) / (float (2.0 *. width *. width)))) in
    (exp_term * (-diff) / (float (width *. width)))

  (* Create a set of Gaussian RBF basis functions *)
  let create_rbf_basis centers width =
    let n = Array.length centers in
    let functions = Array.init n (fun i ->
      fun x -> rbf_gaussian centers.(i) width x
    ) in
    let gradients = Array.init n (fun i ->
      fun x -> rbf_gaussian_gradient centers.(i) width x
    ) in
    create functions gradients
end

(* Measure spaces and importance sampling *)
module type MeasureSpace = sig
  type t
  type path
  type functional
  
  val create : dim:int -> t
  val density : t -> path -> Tensor.t
  val expectation : t -> functional -> Tensor.t
  val relative_entropy : t -> t -> Tensor.t
  val is_absolutely_continuous : t -> t -> bool
end

module PathMeasure = struct
  type t = {
    drift: Tensor.t -> float -> Tensor.t;
    diffusion: Tensor.t -> Tensor.t;
    dim: int;
    time_step: float;
  }

  let create b sigma d dt = {
    drift = b;
    diffusion = sigma;
    dim = d;
    time_step = dt;
  }

  (* Measure transformation via Girsanov *)
  let transform_measure t u_control =
    let drift x time = 
      let open Torch in
      let u = u_control x time in
      let b = t.drift x time in
      let sigma = t.diffusion x in
      b + matmul sigma u
    in
    { t with drift }

  (* Check Novikov condition *)
  let check_novikov t u_control paths =
    let open Torch in
    let n_paths = Array.length paths in
    let n_steps = Array.length paths.(0) in
    
    let compute_path_integral path =
      let integral = ref (zeros [1]) in
      for i = 0 to n_steps - 2 do
        let time = float_of_int i *. t.time_step in
        let u = u_control path.(i) time in
        integral := !integral + dot u u * float t.time_step;
      done;
      float 0.5 * !integral
    in
    
    let expectations = Array.map compute_path_integral paths in
    let exp_sum = 
      stack (Array.to_list expectations) ~dim:0 
      |> exp 
      |> mean 
      |> float_value
    in
    exp_sum < infinity
end

module ImportanceSampling = struct
  type measure = {
    drift: Tensor.t -> float -> Tensor.t;
    diffusion: Tensor.t -> Tensor.t;
    dimension: int;
  }

  type t = {
    reference_measure: measure;
    modified_measure: measure;
    time_step: float;
    terminal_time: float;
  }

  let create ref_measure mod_measure dt t_final = {
    reference_measure = ref_measure;
    modified_measure = mod_measure;
    time_step = dt;
    terminal_time = t_final;
  }

  (* Compute log likelihood ratio *)
  let log_likelihood_ratio t trajectory =
    let open Torch in
    let steps = Array.length trajectory - 1 in
    let dt = t.time_step in
    
    let compute_step_ratio i x_i x_next =
      let time = float_of_int i *. dt in
      let b_ref = t.reference_measure.drift x_i time in
      let b_mod = t.modified_measure.drift x_i time in
      let sigma = t.reference_measure.diffusion x_i in
      let sigma_inv = inverse sigma in
      
      let dx = x_next - x_i in
      let u = (b_mod - b_ref) in
      
      let term1 = dot u (matmul sigma_inv dx) in
      let term2 = -(float 0.5) * dt * dot u (matmul sigma_inv u) in
      term1 + term2
    in
    
    let ratio = ref (zeros [1]) in
    for i = 0 to steps - 1 do
      ratio := !ratio + compute_step_ratio i trajectory.(i) trajectory.(i+1);
    done;
    !ratio

  (* Zero-variance importance sampling estimator *)
  let zero_variance_estimator t paths cost_functional =
    let open Torch in
    let n_paths = Array.length paths in
    
    let compute_path_estimate path =
      let ll_ratio = log_likelihood_ratio t path in
      let cost = cost_functional path in
      exp (ll_ratio - cost)
    in
    
    let estimates = Array.map compute_path_estimate paths in
    let estimate_tensor = stack (Array.to_list estimates) ~dim:0 in
    -log (mean estimate_tensor)
end

module DonskerVaradhan = struct
  type t = {
    reference_measure: PathMeasure.t;
    functional: Tensor.t array -> Tensor.t;
  }

  let create measure func = {
    reference_measure = measure;
    functional = func;
  }

  let compute_inf_convolution t candidate_measures =
    let open Torch in
    let compute_measure_value measure =
      let paths = Array.make 1000 [|zeros [t.reference_measure.dim]|] in
      let value = t.functional paths in
      let entropy = PathMeasure.relative_entropy measure t.reference_measure in
      value + entropy
    in
    
    let values = Array.map compute_measure_value candidate_measures in
    (stack (Array.to_list values) ~dim:0 |> min ~dim:[0] ~keepdim:false)
end

(* Trajectory management and LSMC *)
module TrajectoryManager = struct
  type path = {
    states: Tensor.t array;
    times: float array;
    terminal_idx: int option;
  }

  type t = {
    paths: path array;
    time_grid: float array;
    dim: int;
    num_paths: int;
  }

  let create dim n_paths time_grid = {
    paths = Array.make n_paths {
      states = Array.make (Array.length time_grid) (zeros [dim]);
      times = time_grid;
      terminal_idx = None;
    };
    time_grid = time_grid;
    dim = dim;
    num_paths = n_paths;
  }

  let get_state t path_idx time_idx =
    t.paths.(path_idx).states.(time_idx)

  let set_state t path_idx time_idx state =
    t.paths.(path_idx).states.(time_idx) <- state

  let set_terminal t path_idx term_idx =
    t.paths.(path_idx) <- { t.paths.(path_idx) with terminal_idx = Some term_idx }

  let is_terminated t path_idx time_idx =
    match t.paths.(path_idx).terminal_idx with
    | Some idx -> time_idx >= idx
    | None -> false
end

module LeastSquaresSolvers = struct
  type solver_type = 
    | SVD
    | QR
    | Cholesky

  type t = {
    solver: solver_type;
    batch_size: int;
    threshold: float;
  }

  let create solver_type batch_sz thresh = {
    solver = solver_type;
    batch_size = batch_sz;
    threshold = thresh;
  }

  (* Solve system using SVD *)
  let solve_svd a b =
    let open Torch in
    let u, s, v = svd a ~some:true in
    let s_inv = zeros (size s) in
    for i = 0 to (size s).(0) - 1 do
      let si = get s [i] in
      if float_value si > 1e-10 then
        set s_inv [i] (float 1.0 /. float_value si);
    done;
    matmul (matmul v (diagflat s_inv)) (transpose u ~dim0:0 ~dim1:1) |> matmul b

  (* Solve system using QR *)
  let solve_qr a b =
    let open Torch in
    let q, r = qr a ~some:true in
    let x = triangular_solve r b ~upper:true ~transpose:false in
    matmul (transpose q ~dim0:0 ~dim1:1) x

  (* Solve system using Cholesky *)
  let solve_cholesky a b =
    let open Torch in
    let ata = matmul (transpose a ~dim0:0 ~dim1:1) a in
    let atb = matmul (transpose a ~dim0:0 ~dim1:1) b in
    let l = cholesky ata ~upper:false in
    let y = triangular_solve l atb ~upper:false ~transpose:false in
    triangular_solve l y ~upper:false ~transpose:true

  (* Main solve function with batching *)
  let solve t a b =
    let open Torch in
    let n_rows = (size a).(0) in
    let n_batches = (n_rows + t.batch_size - 1) / t.batch_size in
    
    let results = ref [] in
    for i = 0 to n_batches - 1 do
      let start_idx = i * t.batch_size in
      let end_idx = min (start_idx + t.batch_size) n_rows in
      
      let a_batch = narrow a ~dim:0 ~start:start_idx 
                      ~length:(end_idx - start_idx) in
      let b_batch = narrow b ~dim:0 ~start:start_idx 
                      ~length:(end_idx - start_idx) in
      
      let result = match t.solver with
        | SVD -> solve_svd a_batch b_batch
        | QR -> solve_qr a_batch b_batch
        | Cholesky -> solve_cholesky a_batch b_batch
      in
      results := result :: !results;
    done;
    
    cat (List.rev !results) ~dim:0
end

module AdvancedLSMC = struct
  type config = {
    num_paths: int;
    time_steps: int;
    basis_size: int;
    solver: LeastSquaresSolvers.t;
  }

  type t = {
    config: config;
    basis: BasisFunctions.t;
    paths: TrajectoryManager.t;
  }

  let create cfg basis paths = {
    config = cfg;
    basis = basis;
    paths = paths;
  }

  (* Improved backward iteration with specialized solvers *)
  let backward_iteration t driver terminal_condition =
    let open Torch in
    let n_steps = t.config.time_steps in
    let y_values = Array.make n_steps (zeros [t.config.num_paths]) in
    let z_values = Array.make n_steps (zeros [t.config.num_paths; t.paths.dim]) in
    
    (* Terminal values *)
    for i = 0 to t.config.num_paths - 1 do
      let final_state = TrajectoryManager.get_state t.paths i (n_steps - 1) in
      Tensor.set y_values.(n_steps - 1) [i] (terminal_condition final_state);
    done;
    
    (* Backward iteration *)
    for n = n_steps - 2 downto 0 do
      let time = float_of_int n *. t.paths.time_grid.(1) in
      
      (* Get states at time n *)
      let states = Array.init t.config.num_paths (fun i ->
        TrajectoryManager.get_state t.paths i n) in
      
      (* Assemble regression matrices *)
      let a = zeros [t.config.num_paths; t.config.basis_size] in
      let b = y_values.(n + 1) in
      
      for i = 0 to t.config.num_paths - 1 do
        for j = 0 to t.config.basis_size - 1 do
          let phi_j = BasisFunctions.evaluate t.basis states.(i) j in
          Tensor.set a [i; j] phi_j;
        done;
      done;
      
      (* Solve regression problem *)
      let coeffs = LeastSquaresSolvers.solve t.config.solver a b in
      
      (* Update y and z values *)
      for i = 0 to t.config.num_paths - 1 do
        let x_i = states.(i) in
        
        (* Compute y value *)
        let y_i = ref (zeros [1]) in
        let z_i = ref (zeros [t.paths.dim]) in
        
        for k = 0 to t.config.basis_size - 1 do
          let phi_k = BasisFunctions.evaluate t.basis x_i k in
          let grad_phi_k = BasisFunctions.evaluate_gradient t.basis x_i k in
          let alpha_k = Tensor.get coeffs [k] in
          y_i := !y_i + alpha_k * phi_k;
          z_i := !z_i + alpha_k * grad_phi_k;
        done;
        
        (* Update values with driver term *)
        let dt = t.paths.time_grid.(1) -. t.paths.time_grid.(0) in
        y_i := !y_i + driver x_i !y_i !z_i time * float dt;
        
        Tensor.set y_values.(n) [i] !y_i;
        Tensor.set z_values.(n) [i] !z_i;
      done;
    done;
    
    y_values, z_values
end

(* Stochastic control and value function components *)
module StochasticGenerator = struct
  type t = {
    drift: Tensor.t -> float -> Tensor.t;
    diffusion: Tensor.t -> Tensor.t;
    dim: int;
  }

  let create b sigma d = {
    drift = b;
    diffusion = sigma;
    dim = d;
  }

  (* Generator *)
  let apply t state gradient =
    let open Torch in
    let sigma = t.diffusion state in
    let sigma_sigma_t = matmul sigma (transpose sigma ~dim0:0 ~dim1:1) in
    let drift_term = t.drift state 0.0 in
    
    (* First order term *)
    let first_order = dot drift_term gradient in
    
    (* Second order term *)
    let second_order = float 0.5 * trace (matmul sigma_sigma_t gradient) in
    
    first_order + second_order

  (* Generator for controlled process *)
  let controlled_generator t control state gradient =
    let open Torch in
    let base_gen = apply t state gradient in
    let control_term = dot (t.diffusion state * control) gradient in
    base_gen + control_term
end

module ValueFunctionSolver = struct
  type config = {
    state_dim: int;
    control_dim: int;
    time_step: float;
    terminal_time: float;
    space_steps: int array;
  }

  type grid_point = {
    state: Tensor.t;
    value: Tensor.t;
    gradient: Tensor.t option;
  }

  type t = {
    config: config;
    generator: Tensor.t -> Tensor.t -> Tensor.t;
    running_cost: Tensor.t -> float -> Tensor.t;
    terminal_cost: Tensor.t -> Tensor.t;
  }

  let create cfg gen f g = {
    config = cfg;
    generator = gen;
    running_cost = f;
    terminal_cost = g;
  }

  (* Grid generation for PDE solver *)
  let generate_grid t bounds =
    let open Torch in
    let dim = t.config.state_dim in
    let total_points = Array.fold_left ( * ) 1 t.config.space_steps in
    let grid = Array.make total_points {
      state = zeros [dim];
      value = zeros [1];
      gradient = None;
    } in
    
    let idx_to_coord idx =
      let coords = Array.make dim 0 in
      let mut_idx = ref idx in
      for d = 0 to dim - 1 do
        coords.(d) <- !mut_idx mod t.config.space_steps.(d);
        mut_idx := !mut_idx / t.config.space_steps.(d);
      done;
      coords
    in
    
    for i = 0 to total_points - 1 do
      let coords = idx_to_coord i in
      let state = Array.mapi (fun d c ->
        let step = (bounds.(d).max -. bounds.(d).min) /. 
                  float_of_int (t.config.space_steps.(d) - 1) in
        float (bounds.(d).min +. step *. float_of_int c)
      ) coords |> Array.to_list |> Tensor.of_float1 in
      grid.(i) <- { state; value = zeros [1]; gradient = None };
    done;
    grid

  (* PDE solver using finite differences *)
  let solve_pde t grid =
    let open Torch in
    let n_points = Array.length grid in
    let dt = t.config.time_step in
    let n_steps = int_of_float (t.config.terminal_time /. dt) in
    
    (* Initialize terminal condition *)
    for i = 0 to n_points - 1 do
      grid.(i) <- { grid.(i) with 
        value = t.terminal_cost grid.(i).state 
      };
    done;
    
    (* Backward in time iteration *)
    for step = n_steps - 1 downto 0 do
      let time = float_of_int step *. dt in
      
      (* Update each grid point *)
      for i = 0 to n_points - 1 do
        let point = grid.(i) in
        
        (* Compute spatial derivatives *)
        let gradient = ref (zeros [t.config.state_dim]) in
        let hessian = ref (zeros [t.config.state_dim; t.config.state_dim]) in
        
        (* Central difference for gradient *)
        for d = 0 to t.config.state_dim - 1 do
          let h = t.config.time_step in
          let fwd_state = point.state + (zeros [t.config.state_dim] |> set1 d (float h)) in
          let bwd_state = point.state - (zeros [t.config.state_dim] |> set1 d (float h)) in
          let fwd_val = t.terminal_cost fwd_state in
          let bwd_val = t.terminal_cost bwd_state in
          set !gradient [d] ((fwd_val - bwd_val) / (float (2.0 *. h)));
          
          (* Second derivatives for Hessian *)
          for d2 = 0 to t.config.state_dim - 1 do
            let h2 = t.config.time_step in
            let fwd2_state = fwd_state + 
              (zeros [t.config.state_dim] |> set1 d2 (float h2)) in
            let bwd2_state = bwd_state + 
              (zeros [t.config.state_dim] |> set1 d2 (float h2)) in
            let fwd2_val = t.terminal_cost fwd2_state in
            let bwd2_val = t.terminal_cost bwd2_state in
            let mixed_der = (fwd2_val - fwd_val - bwd2_val + bwd_val) /. 
                          (4.0 *. h *. h2) in
            set !hessian [d; d2] (float mixed_der);
          done;
        done;
        
        (* Update value using PDE *)
        let running = t.running_cost point.state time in
        let gen = t.generator point.state !gradient in
        let diffusion_term = matmul !hessian gen in
        
        let new_value = point.value - 
          (running + diffusion_term) * float dt in
        
        grid.(i) <- { point with 
          value = new_value;
          gradient = Some !gradient 
        };
      done;
    done;
    grid
end

module OptimalControl = struct
  type control_config = {
    state_dim: int;
    control_dim: int;
    time_step: float;
    terminal_time: float;
    max_iter: int;
    tol: float;
  }

  type t = {
    config: control_config;
    value_solver: ValueFunctionSolver.t;
    basis: BasisFunctions.t;
  }

  let create cfg val_solver basis = {
    config = cfg;
    value_solver = val_solver;
    basis = basis;
  }

  (* Compute optimal feedback control *)
  let feedback_control t value_grad state time =
    let open Torch in
    let module F = (val t.value_solver.forward_process : Process) in
    let sigma = F.diffusion state in
    matmul (transpose sigma ~dim0:0 ~dim1:1) value_grad

  (* Controlled process evolution *)
  let controlled_evolution t state control time noise dt =
    let open Torch in
    let module F = (val t.value_solver.forward_process : Process) in
    let drift = F.drift state time in
    let diffusion = F.diffusion state in
    let controlled_drift = drift + matmul diffusion control in
    state + controlled_drift * float dt + 
    matmul diffusion noise * float (sqrt dt)

  (* Policy iteration for control optimization *)
  let policy_iteration t initial_condition terminal_condition running_cost =
    let open Torch in
    let n_steps = t.config.max_iter in
    
    (* Initialize value function approximation *)
    let value_coeff = ref (zeros [Array.length t.basis.functions]) in
    let old_coeff = ref (zeros [Array.length t.basis.functions]) in
    
    let converged = ref false in
    let iter = ref 0 in
    
    while not !converged && !iter < n_steps do
      old_coeff := !value_coeff;
      
      (* Run value iteration *)
      let bounds = Array.make t.config.state_dim 
        {ValueFunctionSolver.min = -5.0; max = 5.0} in
      let grid = ValueFunctionSolver.generate_grid t.value_solver bounds in
      let updated_grid = ValueFunctionSolver.solve_pde t.value_solver grid in
      
      (* Update value coefficients *)
      let a = zeros [Array.length updated_grid; Array.length t.basis.functions] in
      let b = zeros [Array.length updated_grid] in
      
      for i = 0 to Array.length updated_grid - 1 do
        let point = updated_grid.(i) in
        for j = 0 to Array.length t.basis.functions - 1 do
          let phi_j = BasisFunctions.evaluate t.basis point.state j in
          set a [i; j] phi_j;
        done;
        set b [i] point.value;
      done;
      
      value_coeff := LeastSquaresSolvers.solve_qr a b;
      
      (* Check convergence *)
      let diff = norm (!value_coeff - !old_coeff) ~p:2 ~dim:[0] in
      converged := float_value diff < t.config.tol;
      incr iter;
    done;
    
    !value_coeff
end

(* Configuration *)
module Configuration = struct
  type numerical_config = {
    time_step: float;
    terminal_time: float;
    space_steps: int array;
    basis_size: int;
    batch_size: int;
    solver_type: LeastSquaresSolvers.solver_type;
    integrator_type: NumericalIntegrator.integrator_type;
    tol: float;
  }

  type problem_config = {
    state_dim: int;
    control_dim: int;
    num_paths: int;
    use_importance_sampling: bool;
  }

  type t = {
    numerical: numerical_config;
    problem: problem_config;
  }

  let create num prob = {
    numerical = num;
    problem = prob;
  }

  (* Create standard configuration *)
  let standard_config ~state_dim ~control_dim ~num_paths = 
    create
      {
        time_step = 0.01;
        terminal_time = 1.0;
        space_steps = Array.make state_dim 50;
        basis_size = 20;
        batch_size = 100;
        solver_type = LeastSquaresSolvers.QR;
        integrator_type = NumericalIntegrator.RK2Adaptive;
        tol = 1e-6;
      }
      {
        state_dim = state_dim;
        control_dim = control_dim;
        num_paths = num_paths;
        use_importance_sampling = true;
      }
end

module FBSDESystem = struct
  type config = {
    state_dim: int;
    control_dim: int;
    time_step: float;
    terminal_time: float;
    num_paths: int;
    basis_size: int;
    batch_size: int;
  }

  type t = {
    config: config;
    forward_process: (module Process);
    backward_solver: AdvancedLSMC.t;
    value_solver: ValueFunctionSolver.t;
    integrator: NumericalIntegrator.t;
    importance_sampling: ImportanceSampling.t option;
    measure: PathMeasure.t;
  }

  let create cfg fwd bwd val_solver intg imp_sampling measure = {
    config = cfg;
    forward_process = fwd;
    backward_solver = bwd;
    value_solver = val_solver;
    integrator = intg;
    importance_sampling = imp_sampling;
    measure = measure;
  }

  (* Complete system solution *)
  let solve t initial_condition terminal_condition running_cost =
    let open Torch in
    
    (* Forward path simulation with adaptive stepping *)
    let simulate_paths () =
      let module F = (val t.forward_process : Process) in
      let n_steps = int_of_float (t.config.terminal_time /. t.config.time_step) in
      let paths = Array.make t.config.num_paths [||] in
      
      for i = 0 to t.config.num_paths - 1 do
        let state = ref initial_condition in
        let path = ref [|!state|] in
        let time = ref 0.0 in
        
        while !time < t.config.terminal_time do
          match NumericalIntegrator.step t.integrator !state 
                  (fun x t -> F.drift x t)
                  F.diffusion !time with
          | Some next_state ->
              state := next_state;
              path := Array.append !path [|!state|];
              time := !time +. t.config.time_step;
          | None ->
              (* Reduce step size and retry *)
              let new_step = t.config.time_step /. 2.0 in
              let intg = {t.integrator with time_step = new_step} in
              match NumericalIntegrator.step intg !state 
                      (fun x t -> F.drift x t)
                      F.diffusion !time with
              | Some next_state ->
                  state := next_state;
                  path := Array.append !path [|!state|];
                  time := !time +. new_step;
              | None -> 
                  (* Fallback to Euler step *)
                  let noise = randn [t.config.state_dim] in
                  let dx = F.drift !state !time * float new_step +
                          F.diffusion !state * noise * float (sqrt new_step) in
                  state := !state + dx;
                  path := Array.append !path [|!state|];
                  time := !time +. new_step;
        done;
        paths.(i) <- !path;
      done;
      paths
    in
    
    (* Apply importance sampling if enabled *)
    let apply_importance_sampling paths =
      match t.importance_sampling with
      | Some imp ->
          let weights = Array.map (fun path ->
            exp (ImportanceSampling.log_likelihood_ratio imp path)
          ) paths in
          paths, Some weights
      | None -> paths, None
    in
    
    (* Main solution process *)
    let paths = simulate_paths () in
    let weighted_paths, weights = apply_importance_sampling paths in
    
    (* Solve value function *)
    let bounds = Array.init t.config.state_dim (fun _ -> 
      {ValueFunctionSolver.min = -5.0; max = 5.0}) in
    let grid = ValueFunctionSolver.generate_grid t.value_solver bounds in
    let value_grid = ValueFunctionSolver.solve_pde t.value_solver grid in
    
    (* Backward LSMC solution *)
    let driver state y z time =
      let running = running_cost state time in
      let z_term = float 0.5 * dot z z in
      running + z_term
    in
    
    let y_values, z_values = 
      AdvancedLSMC.backward_iteration t.backward_solver driver terminal_condition in
    
    (* Compute final estimates *)
    let compute_weighted_estimate values weights =
      match weights with
      | Some w ->
          let weighted_vals = Array.map2 (fun v w -> v * w) values w in
          let w_tensor = stack (Array.to_list w) ~dim:0 in
          sum (stack (Array.to_list weighted_vals) ~dim:0) / sum w_tensor
      | None ->
          mean (stack (Array.to_list values) ~dim:0)
    in
    
    let value_estimate = compute_weighted_estimate y_values.(0) weights in
    
    {
      value = value_estimate;
      paths = weighted_paths;
      value_function = value_grid;
      y_evolution = y_values;
      z_evolution = z_values;
    }
end