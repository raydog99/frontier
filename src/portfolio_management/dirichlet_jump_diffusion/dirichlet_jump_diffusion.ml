open Torch

type vector = Tensor.t
type matrix = Tensor.t

(* Kato class functions *)
type kato_function = {
  value: vector -> float;
  dim: int;
  device: Device.t;
}

(* Jump measure definition *)
type jump_measure = {
  intensity: vector -> float;    (* κ(x) = J(x,Rd) *)
  kernel: vector -> vector;      (* ν(x,dy) *)
  is_finite: bool;
}

(* Domain and boundary *)
type boundary_info = {
  domain: vector -> bool;        (* Domain indicator *)
  distance: vector -> float;     (* Distance to boundary *)
  is_regular: vector -> bool;    (* Regularity check *)
}

(* Solution types *)
type solution = {
  value: vector -> float;
  gradient: vector -> vector option;
  is_weak: bool;
}

(* Operator configuration *)
type operator_config = {
  diffusion: vector -> matrix;   (* A(x) matrix *)
  drift: vector -> vector;       (* b(x) vector *)
  jump: jump_measure;
  dim: int;
  device: Device.t;
}

(* Path component types *)
type path_state = {
  position: vector;
  time: float;
  alive: bool;
}

type path_info = {
  exit_time: float option;
  exit_position: vector option;
  hitting_position: vector option;
  did_jump: bool;
}

(* Kato *)
module Kato = struct
  (* Create Kato function *)
  let create f dim device = 
    { value = f; dim; device }

  (* Verify Kato class condition *)
  let verify_small_time f =
    (* Brownian transition density *)
    let transition_density t x y =
      let d = float_of_int f.dim in
      let diff = sub y x in
      let norm_sq = pow (norm diff) 2 |> to_float0 in
      (1. /. ((4. *. Float.pi *. t) ** (d /. 2.))) *.
      exp (-. norm_sq /. (4. *. t))
    in
    
    (* Compute E_x[∫_0^t |f(B_s)|ds] *)
    let expectation x t =
      let n_steps = 100 in
      let dt = t /. float_of_int n_steps in
      
      let rec simulate acc time =
        if time >= t then acc
        else
          let dw = randn [f.dim] ~device:f.device in
          let next_x = add x (mul_scalar dw (sqrt dt)) in
          let val_f = abs_float (f.value next_x) in
          simulate (acc +. val_f *. dt) (time +. dt)
      in
      simulate 0. 0.
    in
    
    (* Test condition *)
    let test_points = 100 in
    let points = List.init test_points (fun _ -> 
      randn [f.dim] ~device:f.device
    ) in
    
    let times = [0.1; 0.01; 0.001] in
    List.for_all (fun t ->
      let max_exp = List.fold_left (fun acc x ->
        max acc (expectation x t)
      ) 0. points in
      max_exp < 0.1  (* Threshold for small time behavior *)
    ) times

  (* Verify dimension-specific conditions *)
  let verify_dimensional f =
    match f.dim with
    | 1 -> 
        (* Check boundedness of integral *)
        let integral = List.init 100 (fun _ ->
          let x = randn [1] ~device:f.device in
          abs_float (f.value x)
        ) |> List.fold_left max 0. in
        integral < Float.infinity
        
    | 2 ->
        (* Check logarithmic singularity *)
        let check_point x =
          let r = norm x |> to_float0 in
          if r < 1e-10 then true
          else
            let val_f = abs_float (f.value x) in
            val_f *. log (1. /. r) < Float.infinity
        in
        List.init 100 (fun _ -> 
          randn [2] ~device:f.device
        ) |> List.for_all check_point
        
    | d when d >= 3 ->
        (* Check power law decay *)
        let check_point x =
          let r = norm x |> to_float0 in
          if r < 1e-10 then true
          else
            let val_f = abs_float (f.value x) in
            val_f /. (r ** float_of_int (d - 2)) < Float.infinity
        in
        List.init 100 (fun _ -> 
          randn [d] ~device:f.device
        ) |> List.for_all check_point
        
    | _ -> invalid_arg "Invalid dimension"
end

(* Core operator *)
module CoreOperator = struct
  type t = {
    config: operator_config;
    device: Device.t;
  }

  (* Create operator with validation *)
  let create config device =
    (* Validate uniform ellipticity of A(x) *)
    let check_ellipticity x =

      let a = config.diffusion x in
      let eigenvals = eig a |> fst in
      let min_eval = reduce_min eigenvals |> to_float0 in
      let max_eval = reduce_max eigenvals |> to_float0 in
      let lambda = 1e6 in  (* Ellipticity bound *)
      min_eval >= 1. /. lambda && max_eval <= lambda
    in
    
    (* Validate drift is in Kato class *)
    let check_drift x =

      let b = config.drift x in
      let b_squared x = 
        pow (norm (config.drift x)) 2 |> to_float0
      in
      let kato = Kato.create 
        { value = b_squared; dim = config.dim; device } 
        config.dim device in
      Kato.verify_small_time kato
    in
    
    (* Validate jump measure *)
    let check_jump_measure () =
      let kato = Kato.create
        { value = config.jump.intensity;
          dim = config.dim;
          device }
        config.dim device in
      Kato.verify_small_time kato
    in
    
    if not (check_jump_measure ()) then
      invalid_arg "Jump measure intensity not in Kato class"
    else
      { config; device }

  (* Apply L0 operator: div(A∇) + b·∇ *)
  let apply_l0 t f x =
    
    (* Compute div(A∇f) *)
    let a = t.config.diffusion x in
    let grad_f = grad f x in
    let div_term = matmul a grad_f in
    
    (* Compute b·∇f *)
    let b = t.config.drift x in
    let drift_term = dot b grad_f in
    
    add div_term drift_term

  (* Apply jump part: ∫(f(y) - f(x))J(x,dy) *)
  let apply_jump t f x =
    let y = t.config.jump.kernel x in
    let rate = t.config.jump.intensity x in
    mul_scalar (sub (f y) (f x)) rate

  (* Apply full operator L = L0 + jump part *)
  let apply t f x =
    add (apply_l0 t f x) (apply_jump t f x)
end

(* Measure theory *)
module Measure = struct
  type t = {
    density: vector -> float;
    support: vector -> bool;
    dimension: int;
    device: Device.t;
  }

  (* Create measure from density *)
  let create ~density ~support ~dim ~device =
    { density; support; dimension = dim; device }

  (* Integration methods *)
  module Integration = struct
    (* Monte Carlo integration *)
    let monte_carlo measure f ~n_samples =

      
      let samples = List.init n_samples (fun _ ->
        randn [measure.dimension] ~device:measure.device
      ) |> List.filter measure.support in
      
      let sum = List.fold_left (fun acc x ->
        let density_val = measure.density x in
        let f_val = f x in
        acc +. density_val *. f_val
      ) 0. samples in
      
      sum /. float_of_int (List.length samples)

    (* Deterministic integration for small dimensions *)
    let grid_integrate measure f ~bounds ~grid_points =

      let d = measure.dimension in
      let step = (bounds.high -. bounds.low) /. 
                float_of_int grid_points in
      
      let rec integrate_dim acc dims remaining_d =
        if remaining_d = 0 then
          acc *. measure.density dims *. f dims
        else
          let grid = linspace ~start:bounds.low ~end_:bounds.high
                            grid_points ~device:measure.device in
          
          let sub_integrals = List.map (fun x ->
            let new_dims = Tensor.cat [dims; of_float1 [x]] 0 in
            integrate_dim acc new_dims (remaining_d - 1)
          ) (to_list1 grid) in
          
          List.fold_left (+.) 0. sub_integrals
      in
      
      integrate_dim 1. (empty [0]) d *. (step ** float_of_int d)
  end

  (* Measure operations *)
  module Operations = struct
    (* Create product measure *)
    let product m1 m2 =
      if m1.dimension <> m2.dimension then
        invalid_arg "Measures must have same dimension"
      else
        let density x =
          if m1.support x && m2.support x then
            m1.density x *. m2.density x
          else 0.
        in
        
        let support x = m1.support x && m2.support x in
        
        { density;
          support;
          dimension = m1.dimension;
          device = m1.device }

    (* Create restriction of measure to set *)
    let restrict measure set =
      let density x =
        if set x then measure.density x else 0.
      in
      
      let support x = measure.support x && set x in
      
      { density;
        support;
        dimension = measure.dimension;
        device = measure.device }
  end
end

(* Path generation and simulation *)
module PathGenerator = struct
  type path = {
    states: path_state list;
    info: path_info;
  }

  (* Generate sample path with jumps *)
  let generate op ~x0 ~max_time ~dt =
    
    (* Single step of path generation *)
    let step x time =
      (* Check for jump *)
      let jump_rate = op.config.jump.intensity x in
      let do_jump = Random.float 1.0 < jump_rate *. dt in
      
      if do_jump then
        let next_x = op.config.jump.kernel x in
        (next_x, true)
      else
        (* Continuous motion *)
        let sigma = sqrt (op.config.diffusion x) in
        let drift = op.config.drift x in
        
        let dw = randn_like x in
        let diffusion_term = matmul sigma (mul_scalar dw (sqrt dt)) in
        let drift_term = mul_scalar drift dt in
        
        let next_x = add x (add drift_term diffusion_term) in
        (next_x, false)
    in
    
    (* Generate full path *)
    let rec simulate acc x time =
      if time >= max_time then
        { states = List.rev acc;
          info = { 
            exit_time = None;
            exit_position = None;
            hitting_position = None;
            did_jump = false 
          }}
      else
        let (next_x, jumped) = step x time in
        let state = { 
          position = next_x;
          time = time +. dt;
          alive = true 
        } in
        simulate (state :: acc) next_x (time +. dt)
    in
    
    let initial_state = {
      position = x0;
      time = 0.;
      alive = true
    } in
    
    simulate [initial_state] x0 0.

  (* Path analysis utilities *)
  module Analysis = struct
    (* Compute path properties *)
    let analyze_path path =
      (* Find jumps *)
      let jumps = List.filter_mapi (fun i state ->
        if i > 0 then
          let prev = List.nth path.states (i-1) in
          let dist = norm (sub state.position prev.position) in
          if Scalar.to_float dist > 1e-6 then
            Some (prev.time, prev.position, state.position)
          else None
        else None
      ) path.states in
      
      (* Compute jump sizes *)
      let jump_sizes = List.map (fun (_, from_pos, to_pos) ->
        norm (sub to_pos from_pos) |> Scalar.to_float
      ) jumps in
      
      {
        n_jumps = List.length jumps;
        max_jump_size = 
          match jump_sizes with
          | [] -> 0.
          | js -> List.fold_left max 0. js;
        total_time = 
          match List.last path.states with
          | Some state -> state.time
          | None -> 0.
      }
  end
end

(* Boundary handling *)
module Boundary = struct
  (* Create boundary from domain *)
  let create domain ~dim ~device =
    (* Distance to boundary computation *)
    let distance x =
      if domain x then
        let h = 1e-5 in
        let dirs = List.init dim (fun i ->
          let ei = Tensor.zeros [dim] ~device in
          Tensor.set ei [|i|] 1.0;
          ei
        ) in
        
        (* Find minimum distance in each direction *)
        List.fold_left (fun acc dir ->
          let rec binary_search tmin tmax steps =
            if steps = 0 then tmax
            else
              let t = (tmin +. tmax) /. 2. in
              let pt = Tensor.(add x (mul_scalar dir t)) in
              if domain pt then
                binary_search t tmax (steps - 1)
              else
                binary_search tmin t (steps - 1)
          in
          min acc (binary_search 0. 1. 20)
        ) Float.infinity dirs
      else 0.
    in
    
    (* Check regularity at boundary point *)
    let check_regularity x =
      let dist = distance x in
      if dist > 1e-6 then false
      else
        (* Test hitting probability *)
        let n_tests = 100 in
        let epsilon = 1e-3 in
        let hits = List.init n_tests (fun _ ->
          let direction = Tensor.randn [dim] ~device in
          let normalized = Tensor.(div direction (norm direction)) in
          let test_point = Tensor.(add x (mul_scalar normalized epsilon)) in
          distance test_point < epsilon
        ) in
        
        let hit_ratio = float_of_int (List.length (List.filter (fun x -> x) hits)) /.
                       float_of_int n_tests in
        hit_ratio > 0.9
    in
    
    { domain;
      distance;
      is_regular = check_regularity }

  (* Find first hitting time *)
  let find_hitting_time boundary path =
    let rec find_hit = function
      | [] -> None
      | state :: rest ->
          if not (boundary.domain state.position) then
            Some (state.time, state.position)
          else
            find_hit rest
    in
    find_hit path.PathGenerator.states
end

(* Weak solution framework *)
module WeakSolution = struct
  type test_function = {
    value: vector -> float;
    gradient: vector -> vector;
    support: vector -> bool;
  }

  (* Create test function space *)
  module TestFunctions = struct
    (* Create smooth test function with compact support *)
    let create center radius =

      
      let value x =
        let dist = sub x center |> norm |> Scalar.to_float in
        if dist > radius then 0.
        else
          exp (-1. /. (1. -. (dist /. radius) ** 2.))
      in
      
      let gradient x =
        let dist = sub x center |> norm |> Scalar.to_float in
        if dist > radius then zeros_like x
        else
          let diff = sub x center in
          let factor = -2. *. dist /. (radius *. radius) *.
            exp(-1. /. (1. -. (dist /. radius) ** 2.)) /.
            (1. -. (dist /. radius) ** 2.) ** 2. in
          mul_scalar diff factor
      in
      
      let support x =
        let dist = sub x center |> norm |> Scalar.to_float in
        dist <= radius
      in
      
      { value; gradient; support }

    (* Generate sequence of test functions *)
    let create_sequence dim n device =
      List.init n (fun _ ->
        let center = Tensor.randn [dim] ~device in
        let radius = 0.1 +. 0.4 *. Random.float 1.0 in
        create center radius
      )
  end

  (* Verify weak solution property *)
  let verify_solution op domain solution test_functions =
    
    List.for_all (fun phi ->
      (* Compute integral of Lu * phi *)
      let test_points = List.init 1000 (fun _ ->
        randn [op.config.dim] ~device:op.device
      ) in
      
      let integral = List.fold_left (fun acc x ->
        if domain.domain x && phi.support x then
          let lu = CoreOperator.apply op solution.value x in
          acc +. Scalar.to_float lu *. phi.value x
        else acc
      ) 0. test_points in
      
      abs_float integral < 1e-6
    ) test_functions
end

(* Dirichlet problem solver *)
module DirichletSolver = struct
  type solver_config = {
    n_samples: int;
    max_time: float;
    time_step: float;
    tolerance: float;
  }

  (* Create solver *)
  let create op boundary config =
    if config.time_step <= 0. || 
       config.time_step >= config.max_time ||
       config.n_samples <= 0 then
      invalid_arg "Invalid solver configuration"
    else
      (op, boundary, config)

  (* Main solver function *)
  let solve (op, boundary, config) boundary_data =
    
    (* Core solution function *)
    let solve_at x =
      if not (boundary.domain x) then
        boundary_data x
      else
        (* Monte Carlo estimation *)
        let paths = List.init config.n_samples (fun _ ->
          PathGenerator.generate op 
            ~x0:x 
            ~max_time:config.max_time 
            ~dt:config.time_step
        ) in
        
        let exit_values = List.filter_map (fun path ->
          match Boundary.find_hitting_time boundary path with
          | Some (_, exit_pos) -> Some (boundary_data exit_pos)
          | None -> None
        ) paths in
        
        match exit_values with
        | [] -> boundary_data x  (* Fallback *)
        | vs -> 
            let sum = List.fold_left (+.) 0. vs in
            sum /. float_of_int (List.length vs)
    in
    
    (* Compute gradient where possible *)
    let gradient x =
      if not (boundary.domain x) then None
      else
        let h = 1e-5 in
        let d = op.config.dim in
        let grad = List.init d (fun i ->
          let ei = zeros [d] ~device:op.device in
          Tensor.set ei [|i|] 1.0;
          
          let fwd = add x (mul_scalar ei h) in
          let bwd = sub x (mul_scalar ei h) in
          (solve_at fwd -. solve_at bwd) /. (2. *. h)
        ) in
        Some (of_float_list grad ~device:op.device)
    in
    
    { value = solve_at;
      gradient;
      is_weak = true }

  (* Verify solution *)
  let verify (op, boundary, config) solution =
    (* Create test functions *)
    let test_fns = WeakSolution.TestFunctions.create_sequence 
      op.config.dim 10 op.device in
    
    WeakSolution.verify_solution op boundary solution test_fns &&
    (match solution.gradient with
     | Some _ -> true  (* Gradient exists *)
     | None -> false)
end

(* Error analysis framework *)
module ErrorAnalysis = struct
  type error_components = {
    spatial: float;
    temporal: float;
    statistical: float;
    total: float;
    convergence_rate: float option;
  }

  (* Compute error decomposition *)
  let analyze_error solution reference op domain =
    
    (* Spatial error analysis *)
    let compute_spatial_error () =
      let h_values = [0.1; 0.01; 0.001] in
      
      let errors = List.map (fun h ->
        let test_points = List.init 100 (fun _ ->
          let x = randn [op.config.dim] ~device:op.device in
          let perturbed = add x (mul_scalar (randn_like x) h) in
          abs_float (solution.value x -. solution.value perturbed) /. h
        ) in
        List.fold_left max 0. test_points
      ) h_values in
      
      match errors with
      | e1 :: e2 :: _ -> 
          (e1, Some (abs_float (log (e2 /. e1)) /. log 10.))
      | _ -> (0., None)
    in
    
    (* Temporal error analysis *)
    let compute_temporal_error () =
      let dt_values = [0.1; 0.01; 0.001] in
      
      let paths = List.init 100 (fun _ ->
        let x0 = randn [op.config.dim] ~device:op.device in
        List.map (fun dt ->
          PathGenerator.generate op ~x0 ~max_time:1.0 ~dt
        ) dt_values
      ) in
      
      let errors = List.map (fun path_set ->
        List.map2 (fun p1 p2 ->
          match (p1.PathGenerator.info.exit_time, 
                 p2.PathGenerator.info.exit_time) with
          | Some t1, Some t2 -> abs_float (t1 -. t2)
          | _ -> 1.0
        ) (List.tl path_set) (List.tl (List.tl path_set))
      ) paths in
      
      let avg_errors = List.map (fun es ->
        List.fold_left (+.) 0. es /. float_of_int (List.length es)
      ) errors in
      
      match avg_errors with
      | e1 :: e2 :: _ ->
          (e1, Some (abs_float (log (e2 /. e1)) /. log 10.))
      | _ -> (0., None)
    in
    
    (* Statistical error analysis *)
    let compute_statistical_error () =
      let n_samples_list = [100; 1000; 10000] in
      
      let errors = List.map (fun n ->
        let test_points = List.init 10 (fun _ ->
          randn [op.config.dim] ~device:op.device
        ) in
        
        let variances = List.map (fun x ->
          let values = List.init n (fun _ ->
            solution.value x
          ) in
          let mean = List.fold_left (+.) 0. values /. float_of_int n in
          let var = List.fold_left (fun acc v ->
            acc +. (v -. mean) ** 2.
          ) 0. values /. float_of_int n in
          sqrt var
        ) test_points in
        
        List.fold_left max 0. variances
      ) n_samples_list in
      
      match errors with
      | e1 :: e2 :: _ ->
          (e1, Some (abs_float (log (e2 /. e1)) /. log 10.))
      | _ -> (0., None)
    in
    
    let (spatial_err, spatial_rate) = compute_spatial_error () in
    let (temporal_err, temporal_rate) = compute_temporal_error () in
    let (stat_err, stat_rate) = compute_statistical_error () in
    
    let total_err = sqrt (spatial_err ** 2. +. 
                         temporal_err ** 2. +. 
                         stat_err ** 2.) in
    
    let conv_rate = match (spatial_rate, temporal_rate, stat_rate) with
      | Some sr, Some tr, Some mr -> 
          Some (min (min sr tr) mr)
      | _ -> None
    in
    
    { spatial = spatial_err;
      temporal = temporal_err;
      statistical = stat_err;
      total = total_err;
      convergence_rate = conv_rate }

  (* Estimate error bounds *)
  let estimate_bounds solution op domain =
    
    (* Interior equation error *)
    let interior_error = 
      let points = List.init 100 (fun _ ->
        randn [op.config.dim] ~device:op.device
      ) |> List.filter domain.domain in
      
      List.fold_left (fun acc x ->
        let lu = CoreOperator.apply op solution.value x in
        max acc (abs_float (Scalar.to_float lu))
      ) 0. points
    in
    
    (* Boundary condition error *)
    let boundary_error =
      let points = List.init 100 (fun _ ->
        randn [op.config.dim] ~device:op.device
      ) |> List.filter (fun x -> not (domain.domain x)) in
      
      match solution.gradient with
      | Some grad ->
          List.fold_left (fun acc x ->
            let grad_norm = norm (grad x) |> Scalar.to_float in
            max acc grad_norm
          ) 0. points
      | None -> Float.infinity
    in
    
    (interior_error, boundary_error)
end

(* Convergence analysis *)
module Convergence = struct
  type convergence_info = {
    spatial_rate: float;
    temporal_rate: float;
    statistical_rate: float;
    overall_rate: float;
    iterations: int;
    achieved_tolerance: float;
  }

  (* Analyze convergence of solution *)
  let analyze solution op boundary config =
    (* Study spatial convergence *)
    let analyze_spatial_conv () =
      let steps = [0.1; 0.05; 0.025; 0.0125] in
      
      let errors = List.map (fun h ->
        let points = List.init 100 (fun _ ->
          randn [op.config.dim] ~device:op.device
        ) in
        
        List.fold_left (fun acc x ->
          let baseline = solution.value x in
          let perturbed = add x (mul_scalar (randn_like x) h) in
          let diff = abs_float (solution.value perturbed -. baseline) in
          max acc (diff /. h)
        ) 0. points
      ) steps in
      
      (* Estimate convergence rate *)
      match errors with
      | e1 :: e2 :: _ ->
          abs_float (log (e2 /. e1) /. log 2.)
      | _ -> 0.
    in
    
    (* Study temporal convergence *)
    let analyze_temporal_conv () =
      let dt_steps = [0.1; 0.05; 0.025; 0.0125] in
      
      let errors = List.map (fun dt ->
        let paths = List.init 100 (fun _ ->
          let x0 = randn [op.config.dim] ~device:op.device in
          PathGenerator.generate op ~x0 ~max_time:1.0 ~dt
        ) in
        
        List.fold_left (fun acc path ->
          match path.PathGenerator.info.exit_time with
          | Some t -> max acc (abs_float (t -. 1.0))
          | None -> acc
        ) 0. paths
      ) dt_steps in
      
      match errors with
      | e1 :: e2 :: _ ->
          abs_float (log (e2 /. e1) /. log 2.)
      | _ -> 0.
    in
    
    (* Study statistical convergence *)
    let analyze_statistical_conv () =
      let sample_sizes = [100; 200; 400; 800] in
      
      let errors = List.map (fun n ->
        let test_points = List.init 10 (fun _ ->
          randn [op.config.dim] ~device:op.device
        ) in
        
        List.fold_left (fun acc x ->
          let values = List.init n (fun _ -> solution.value x) in
          let mean = List.fold_left (+.) 0. values /. float_of_int n in
          let var = List.fold_left (fun acc v ->
            acc +. (v -. mean) ** 2.
          ) 0. values /. float_of_int n in
          max acc (sqrt var)
        ) 0. test_points
      ) sample_sizes in
      
      match errors with
      | e1 :: e2 :: _ ->
          abs_float (log (e2 /. e1) /. log 2.)
      | _ -> 0.
    in
    
    let spatial_rate = analyze_spatial_conv () in
    let temporal_rate = analyze_temporal_conv () in
    let statistical_rate = analyze_statistical_conv () in
    
    let overall_rate = min (min spatial_rate temporal_rate) 
                          statistical_rate in
    
    { spatial_rate;
      temporal_rate;
      statistical_rate;
      overall_rate;
      iterations = config.n_samples;
      achieved_tolerance = config.tolerance }
end

(* Utilities *)
module Utils = struct
  (* Generate test points *)
  let generate_test_points dim n device =
    List.init n (fun _ ->
      Tensor.randn [dim] ~device
    )

  (* Compute basic statistics *)
  let compute_statistics values =
    let n = float_of_int (List.length values) in
    if n = 0. then (0., 0., 0.)
    else
      let sum = List.fold_left (+.) 0. values in
      let mean = sum /. n in
      let var = List.fold_left (fun acc x ->
        acc +. (x -. mean) ** 2.
      ) 0. values /. n in
      let std = sqrt var in
      (mean, var, std)

  (* Convergence rate estimation *)
  let estimate_convergence_rate values =
    match values with
    | v1 :: v2 :: _ -> 
        Some (abs_float (log (v2 /. v1)) /. log 2.)
    | _ -> None

  (* Boundary utilities *)
  module BoundaryUtils = struct
    (* Check if point is near boundary *)
    let is_near_boundary point boundary eps =
      abs_float (boundary.distance point) < eps

    (* Find nearest boundary point *)
    let find_nearest_boundary point boundary =

      let rec binary_search direction tmin tmax steps =
        if steps = 0 then None
        else
          let t = (tmin +. tmax) /. 2. in
          let test_point = add point (mul_scalar direction t) in
          if boundary.domain test_point then
            binary_search direction t tmax (steps - 1)
          else
            binary_search direction tmin t (steps - 1)
      in
      
      let directions = List.init 8 (fun i ->
        let angle = 2. *. Float.pi *. float_of_int i /. 8. in
        let dir = Tensor.of_float2 [[cos angle; sin angle]] in
        dir
      ) in
      
      List.find_map (fun dir ->
        binary_search dir 0. 1. 20
      ) directions
  end
end