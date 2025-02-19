open Torch

type state = Tensor.t
type control = Tensor.t
type time = float
type measure = Tensor.t

let beta = 0.1 (* discount rate *)
let r = 1.0 (* bound for compact support *)
let k = 1.0 (* bound for control space *)

(* Measure space *)
module Measure = struct
  (* Complete measure type with all required properties *)
  type t = {
    density: Tensor.t;
    support: float * float;
    grid_points: int;
  }

  (* Create measure ensuring proper normalization *)
  let create density support grid_points =
    let normalized = Tensor.div density (Tensor.sum density) in
    {density = normalized; support; grid_points}

  (* Measure topology *)
  module Topology = struct
    (* Weak convergence *)
    let check_weak_convergence measure1 measure2 test_functions =
      let max_diff = ref 0.0 in
      Array.iter (fun f ->
        let exp1 = Tensor.dot measure1.density (f measure1.grid_points) in
        let exp2 = Tensor.dot measure2.density (f measure2.grid_points) in
        let diff = abs_float (Tensor.item (Tensor.sub exp1 exp2)) in
        max_diff := max !max_diff diff
      ) test_functions;
      !max_diff

    (* Total variation distance *)
    let total_variation measure1 measure2 =
      let diff = Tensor.sub measure1.density measure2.density in
      Tensor.abs diff |> Tensor.sum |> Tensor.item |> (fun x -> x /. 2.0)

    (* Wasserstein distance *)
    let wasserstein_distance measure1 measure2 =
      let transport_plan = ref (Tensor.zeros [measure1.grid_points; measure2.grid_points]) in
      let cost = ref Float.infinity in
      for iter = 1 to 100 do
        let proposed_plan = Tensor.randn [measure1.grid_points; measure2.grid_points] in
        let normalized_plan = Tensor.div proposed_plan (Tensor.sum proposed_plan) in
        let marginal1 = Tensor.sum normalized_plan ~dim:[1] in
        let marginal2 = Tensor.sum normalized_plan ~dim:[0] in
        if Tensor.allclose marginal1 measure1.density &&
           Tensor.allclose marginal2 measure2.density then
          let current_cost = Tensor.sum (Tensor.mul normalized_plan 
            (Tensor.abs (Tensor.sub measure1.density measure2.density))) 
            |> Tensor.item in
          if current_cost < !cost then begin
            cost := current_cost;
            transport_plan := normalized_plan
          end
      done;
      !cost
  end

  (* Measure derivatives *)
  module Derivatives = struct
    (* First variation derivative *)
    let first_variation measure phi =
      let eps = 1e-6 in
      let n = measure.grid_points in
      let derivative = Tensor.zeros [n] in
      for i = 0 to n - 1 do
        let perturbed = Tensor.copy measure.density in
        Tensor.set perturbed [|i|] (Tensor.get measure.density [|i|] +. eps);
        let perturbed_measure = {measure with density = perturbed} in
        let diff = (phi perturbed_measure -. phi measure) /. eps in
        Tensor.set derivative [|i|] diff
      done;
      derivative

    (* Second variation derivative *)
    let second_variation measure phi =
      let eps = 1e-6 in
      let n = measure.grid_points in
      let hessian = Tensor.zeros [n; n] in
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let perturbed_i = Tensor.copy measure.density in
          let perturbed_j = Tensor.copy measure.density in
          let perturbed_ij = Tensor.copy measure.density in
          
          Tensor.set perturbed_i [|i|] (Tensor.get measure.density [|i|] +. eps);
          Tensor.set perturbed_j [|j|] (Tensor.get measure.density [|j|] +. eps);
          Tensor.set perturbed_ij [|i|] (Tensor.get measure.density [|i|] +. eps);
          Tensor.set perturbed_ij [|j|] (Tensor.get measure.density [|j|] +. eps);
          
          let measure_i = {measure with density = perturbed_i} in
          let measure_j = {measure with density = perturbed_j} in
          let measure_ij = {measure with density = perturbed_ij} in
          
          let diff = (phi measure_ij -. phi measure_i -. phi measure_j +. phi measure) 
                    /. (eps *. eps) in
          Tensor.set hessian [|i; j|] diff
        done
      done;
      hessian

    (* Lions derivative *)
    let lions_derivative measure phi =
      let first = first_variation measure phi in
      let second = second_variation measure phi in
      (first, second)
  end

  (* Measure transformations *)
  module Transform = struct
    (* Push-forward measure *)
    let pushforward measure f =
      let grid = Tensor.linspace 
        (fst measure.support) 
        (snd measure.support) 
        measure.grid_points in
      let transformed_points = Tensor.map f grid in
      let new_support = (
        Tensor.min transformed_points |> Tensor.item,
        Tensor.max transformed_points |> Tensor.item
      ) in
      let new_density = Tensor.zeros [measure.grid_points] in
      for i = 0 to measure.grid_points - 1 do
        let point = Tensor.get grid [|i|] in
        let transformed = f (Tensor.float_tensor [|point|]) in
        let idx = int_of_float (
          (Tensor.item transformed -. fst new_support) *. 
          float_of_int measure.grid_points /. 
          (snd new_support -. fst new_support)
        ) in
        let idx = max 0 (min (measure.grid_points - 1) idx) in
        let current = Tensor.get new_density [|idx|] in
        Tensor.set new_density [|idx|] 
          (current +. Tensor.get measure.density [|i|])
      done;
      {measure with density = new_density; support = new_support}

    (* Check measure regularity *)
    let check_regularity measure =
      let normalized = Tensor.div measure.density
        (Tensor.sum measure.density) in
      Tensor.allclose measure.density normalized
  end
end

(* Stochastic calculus *)
module Stochastic = struct
  (* Filtration with complete structure *)
  type filtration = {
    time_grid: float array;
    events: Tensor.t array;  (* Represents σ-algebras *)
    is_complete: bool;
    is_right_continuous: bool;
  }

  (* Stochastic process *)
  type 'a process = {
    paths: 'a array;
    filtration: filtration;
    time_index: int array;
  }

  (* Create complete filtration *)
  let create_complete_filtration time_grid =
    let n = Array.length time_grid in
    let events = Array.init n (fun i ->
      Tensor.ones [i + 1]
    ) in
    {
      time_grid;
      events;
      is_complete = true;
      is_right_continuous = true;
    }

  (* Stochastic integration *)
  module Integration = struct
    (* Quadratic variation process *)
    let quadratic_variation process =
      let n = Array.length process.paths - 1 in
      Array.init n (fun i ->
        let increment = Tensor.sub 
          process.paths.(i + 1) process.paths.(i) in
        Tensor.mul increment increment
      )

    (* Stochastic integral using Ito formula *)
    let ito_integral integrand process dt =
      let n = Array.length process.paths - 1 in
      let integral = Array.make n (Tensor.zeros []) in
      for i = 0 to n - 1 do
        let increment = Tensor.sub 
          process.paths.(i + 1) process.paths.(i) in
        let value = Tensor.mul 
          integrand.(i) increment in
        integral.(i) <- value
      done;
      integral

    (* Stratonovich integral *)
    let stratonovich_integral integrand process dt =
      let n = Array.length process.paths - 1 in
      let integral = Array.make n (Tensor.zeros []) in
      for i = 0 to n - 1 do
        let increment = Tensor.sub 
          process.paths.(i + 1) process.paths.(i) in
        let midpoint_integrand = 
          Tensor.add integrand.(i) integrand.(i + 1) |>
          Tensor.mul_float 0.5 in
        let value = Tensor.mul 
          midpoint_integrand increment in
        integral.(i) <- value
      done;
      integral
  end

  (* Martingale verification *)
  module Martingale = struct
    (* Check martingale property *)
    let is_martingale process =
      let n = Array.length process.paths - 1 in
      let is_mart = ref true in
      for i = 0 to n - 1 do
        let current = process.paths.(i) in
        let future_values = Array.sub process.paths (i + 1) (n - i) in
        let expectation = Array.fold_left (fun acc x ->
          Tensor.add acc x
        ) (Tensor.zeros []) future_values in
        let expectation = Tensor.div_float expectation 
          (float_of_int (Array.length future_values)) in
        if not (Tensor.allclose current expectation) then
          is_mart := false
      done;
      !is_mart
  end

  (* Adaptedness verification *)
  module Adaptedness = struct
    (* Check if process is adapted *)
    let is_adapted process =
      Array.for_all2 (fun value events ->
        let dim_value = (Tensor.shape value).(0) in
        let dim_events = (Tensor.shape events).(0) in
        dim_value <= dim_events
      ) process.paths process.filtration.events

    (* Check progressive measurability *)
    let is_progressively_measurable process =
      let n = Array.length process.paths in
      let is_prog = ref true in
      for t = 0 to n - 1 do
        let restricted_paths = Array.sub process.paths 0 (t + 1) in
        let restricted_events = Array.sub 
          process.filtration.events 0 (t + 1) in
        if not (is_adapted {
          process with
          paths = restricted_paths;
          filtration = {
            process.filtration with
            events = restricted_events;
          }
        }) then
          is_prog := false
      done;
      !is_prog

    (* Make process adapted *)
    let adapt process =
      let n = Array.length process.paths in
      let adapted_paths = Array.make n (Tensor.zeros []) in
      for i = 0 to n - 1 do
        let events = process.filtration.events.(i) in
        let projection = Tensor.mul 
          process.paths.(i) events in
        adapted_paths.(i) <- projection
      done;
      {process with paths = adapted_paths}
  end

  (* Predictable processes *)
  module Predictable = struct
    (* Check if process is predictable *)
    let is_predictable process =
      let n = Array.length process.paths in
      let pred_paths = Array.make n (Tensor.zeros []) in
      for i = 1 to n - 1 do
        pred_paths.(i) <- process.paths.(i - 1)
      done;
      pred_paths.(0) <- process.paths.(0);
      let pred_process = {process with paths = pred_paths} in
      Adaptedness.is_adapted pred_process
  end
end

(* Control space *)
module Control = struct  
  (* Control with infinite series representation *)
  type infinite_sequence = {
    coefficients: Tensor.t;   (* (vi) sequence *)
    r_values: Tensor.t;       (* (Ri) sequence *)
    truncation_index: int;          (* Current truncation level *)
  }
  
  (* Control in product topology *)
  type control = {
    sequence: infinite_sequence;
    bound: float;                   (* K value *)
    support: float * float;         (* Control space support *)
  }
  
  (* Create infinite sequence with proper truncation *)
  let create_sequence coeffs r_vals trunc_idx =
    if (Tensor.shape coeffs).(0) >= trunc_idx then
      Some {
        coefficients = Tensor.narrow coeffs 0 0 trunc_idx;
        r_values = Tensor.narrow r_vals 0 0 trunc_idx;
        truncation_index = trunc_idx;
      }
    else None

  (* Create control ensuring Σ(Rivi)² ≤ K *)
  let create_control sequence bound support =
    let weighted = Tensor.mul 
      sequence.coefficients sequence.r_values in
    let sum_squares = Tensor.dot weighted weighted in
    if Tensor.item sum_squares <= bound then
      Some {sequence; bound; support}
    else None

  (* Control evaluation *)
  module Evaluation = struct
    (* Evaluate h(v,x) with infinite series *)
    let evaluate_h ctrl x =
      let n = ctrl.sequence.truncation_index in
      let powers = Array.init n (fun i ->
        Tensor.pow x (float_of_int i)
      ) in
      let terms = Array.mapi (fun i power ->
        let coeff = Tensor.narrow 
          ctrl.sequence.coefficients 0 i 1 in
        Tensor.mul coeff power
      ) powers in
      Array.fold_left Tensor.add 
        (Tensor.zeros []) terms

    (* Evaluate derivative of h(v,x) *)
    let evaluate_derivative ctrl x =
      let n = ctrl.sequence.truncation_index in
      let powers = Array.init (n-1) (fun i ->
        let power = float_of_int (i + 1) in
        Tensor.pow x power
      ) in
      let terms = Array.mapi (fun i power ->
        let coeff = Tensor.narrow 
          ctrl.sequence.coefficients 0 (i+1) 1 in
        let factor = float_of_int (i + 1) in
        Tensor.mul_float (Tensor.mul coeff power) factor
      ) powers in
      Array.fold_left Tensor.add 
        (Tensor.zeros []) terms
  end

  (* Control topology *)
  module Topology = struct
    (* Distance in product topology *)
    let product_distance ctrl1 ctrl2 =
      let common_len = min 
        ctrl1.sequence.truncation_index
        ctrl2.sequence.truncation_index in
      let coeffs1 = Tensor.narrow 
        ctrl1.sequence.coefficients 0 0 common_len in
      let coeffs2 = Tensor.narrow 
        ctrl2.sequence.coefficients 0 0 common_len in
      let diff = Tensor.sub coeffs1 coeffs2 in
      Tensor.norm diff 2.0

    (* Check convergence in product topology *)
    let check_convergence sequence =
      let n = Array.length sequence - 1 in
      let distances = Array.init n (fun i ->
        product_distance sequence.(i) sequence.(i+1)
      ) in
      let is_cauchy = ref true in
      for i = 0 to n - 2 do
        if Tensor.item distances.(i) > 
           Tensor.item distances.(i+1) *. 2.0 then
          is_cauchy := false
      done;
      (!is_cauchy, Array.fold_left (fun acc x -> 
        min acc (Tensor.item x)) Float.infinity distances)
  end
end

(* Measure-valued martingale *)
module MVM = struct  
  (* MVM state *)
  type mvm_state = {
    measure: t;
    filtration: Stochastic.filtration;
    time: float;
    path: t array;
  }
  
  (* Create MVM process *)
  let create measure filtration =
    {
      measure;
      filtration;
      time = 0.0;
      path = [|measure|];
    }
    
  (* MVM evolution *)
  module Evolution = struct
    (* Evolve MVM using Kushner-Stratonovich equation *)
    let evolve state control obs dt =
      let h = Control.Evaluation.evaluate_h control in
      let h_deriv = Control.Evaluation.evaluate_derivative control in
      
      (* Innovation term *)
      let normalized_density = state.measure.density in
      let innovation = Tensor.sub obs 
        (Tensor.dot normalized_density 
          (h (Tensor.float_tensor [|state.time|]))) in
          
      (* Measure update *)
      let update_term = Tensor.mul normalized_density
        (Tensor.mul 
          (h (Tensor.float_tensor [|state.time|])) innovation) in
          
      (* Normalization term *)
      let norm_term = Tensor.dot normalized_density
        (h (Tensor.float_tensor [|state.time|])) in
      let norm_update = Tensor.mul normalized_density norm_term in
      
      (* Combined update *)
      let new_density = Tensor.add normalized_density
        (Tensor.sub update_term norm_update) in
        
      (* Normalize *)
      let new_normalized = Tensor.div new_density 
        (Tensor.sum new_density) in
        
      let new_measure = {state.measure with density = new_normalized} in
      let new_path = Array.append state.path [|new_measure|] in
      
      {state with 
        measure = new_measure;
        time = state.time +. dt;
        path = new_path}
        
    (* Verify martingale property *)
    let verify_martingale state test_function =
      let n = Array.length state.path - 1 in
      let diffs = Array.init n (fun i ->
        let current = Transform.pushforward state.path.(i) test_function in
        let next = Transform.pushforward state.path.(i+1) test_function in
        Topology.total_variation current next
      ) in
      Array.fold_left (+.) 0.0 diffs < 1e-6
  end

  (* MVM properties verification *)
  module Properties = struct
    (* Verify measure-valued martingale properties *)
    let verify_properties state =
      let adaptedness = Stochastic.Adaptedness.is_adapted {
        paths = Array.map (fun m -> m.density) state.path;
        filtration = state.filtration;
        time_index = Array.init (Array.length state.path) (fun i -> i);
      } in
      
      let continuity = ref true in
      for i = 0 to Array.length state.path - 2 do
        let d = Topology.wasserstein_distance 
          state.path.(i) state.path.(i+1) in
        if d > 1e-6 then
          continuity := false
      done;
      
      (adaptedness, !continuity)
  end
end

(* HJB equation *)
module HJB = struct  
  (* HJB solution type *)
  type value_function = {
    value: t -> float;
    gradient: t -> Tensor.t;
    hessian: t -> Tensor.t Tensor.t;
  }
  
  type hjb_solution = {
    value_function: value_function;
    optimal_control: t -> Control.control option;
    verification: t -> bool;
  }
  
  (* Hamiltonian computation *)
  module Hamiltonian = struct
    (* Compute σ(v,μ;dx) *)
    let compute_sigma measure control =
      let h = Control.Evaluation.evaluate_h control in
      let conditional_exp = Tensor.dot measure.density
        (h (Tensor.float_tensor [|0.0|])) in
      Tensor.sub 
        (h (Tensor.float_tensor [|0.0|])) 
        (Tensor.float_tensor [|Tensor.item conditional_exp|])

    (* Compute H(μ,r,φ) *)
    let compute measure value grad hess control =
      let h = Control.Evaluation.evaluate_h control in
      let sigma = compute_sigma measure control in
      
      let cost = Cost.evaluate measure control.sequence.coefficients in
      let diff_term = Tensor.dot grad sigma in
      let hess_term = Tensor.dot 
        (Tensor.mul sigma sigma) hess in
      
      beta *. value -. Tensor.item cost -. 
        Tensor.item diff_term -. 
        Tensor.item hess_term /. 2.0
        
    (* Optimize Hamiltonian *)
    let optimize measure value grad hess =
      let best_value = ref Float.infinity in
      let best_control = ref None in
      
      for _ = 1 to 100 do
        let coeffs = Tensor.randn [10] in
        let r_vals = Tensor.ones [10] in
        let seq_opt = Control.create_sequence coeffs r_vals 10 in
        match seq_opt with
        | Some seq ->
            let ctrl_opt = Control.create_control 
              seq k (-.r, r) in
            begin match ctrl_opt with
            | Some ctrl ->
                let ham = compute measure value grad hess ctrl in
                if ham < !best_value then begin
                  best_value := ham;
                  best_control := Some ctrl
                end
            | None -> ()
            end
        | None -> ()
      done;
      (!best_value, !best_control)
  end
  
  (* Viscosity solution verification *)
  module Viscosity = struct
    (* Test function type *)
    type test_function = {
      phi: t -> float;
      grad_phi: t -> Tensor.t;
      hess_phi: t -> Tensor.t Tensor.t;
    }
    
    (* Verify subsolution property *)
    let verify_subsolution value_fn test_fn measure =
      let (ham_value, _) = Hamiltonian.optimize measure
        (test_fn.phi measure)
        (test_fn.grad_phi measure)
        (test_fn.hess_phi measure) in
      ham_value <= 0.0
      
    (* Verify supersolution property *)
    let verify_supersolution value_fn test_fn measure =
      let (ham_value, _) = Hamiltonian.optimize measure
        (test_fn.phi measure)
        (test_fn.grad_phi measure)
        (test_fn.hess_phi measure) in
      ham_value >= 0.0
      
    (* Create test functions *)
    let create_test_functions measure =
      Array.init 10 (fun i ->
        let center = Tensor.add measure.density
          (Tensor.mul_float (Tensor.randn (Tensor.shape measure.density)) 0.1) in
        let scale = 1.0 +. float_of_int i *. 0.1 in
        {
          phi = (fun m -> 
            let diff = Topology.wasserstein_distance m {measure with density = center} in
            scale *. diff *. diff /. 2.0);
          grad_phi = (fun m ->
            let diff = Tensor.sub m.density center in
            Tensor.mul_float diff scale);
          hess_phi = (fun m ->
            let n = (Tensor.shape m.density).(0) in
            Tensor.eye n |>
            Tensor.mul_float scale);
        }
      )
  end
  
  (* Solver *)
  let solve measure dt =
    (* Initialize value function *)
    let value_fn = {
      value = (fun _ -> 0.0);
      gradient = (fun _ -> Tensor.zeros [measure.grid_points]);
      hessian = (fun _ -> Tensor.zeros [measure.grid_points; measure.grid_points]);
    } in
    
    (* Policy iteration *)
    let rec iterate value_fn iter max_iter =
      if iter >= max_iter then
        value_fn
      else
        let (_, optimal_control) = Hamiltonian.optimize measure
          (value_fn.value measure)
          (value_fn.gradient measure)
          (value_fn.hessian measure) in
          
        match optimal_control with
        | Some ctrl ->
            let new_value = Cost.evaluate measure ctrl.sequence.coefficients in
            let new_grad = Derivatives.first_variation measure 
              (fun _ -> Tensor.item new_value) in
            let new_hess = Derivatives.second_variation measure
              (fun _ -> Tensor.item new_value) in
              
            let new_value_fn = {
              value = (fun m -> Tensor.item new_value);
              gradient = (fun m -> new_grad);
              hessian = (fun m -> new_hess);
            } in
            
            iterate new_value_fn (iter + 1) max_iter
        | None -> value_fn
    in
    
    let final_value_fn = iterate value_fn 0 100 in
    
    (* Compute optimal control *)
    let optimal_control m = 
      let (_, ctrl) = Hamiltonian.optimize m
        (final_value_fn.value m)
        (final_value_fn.gradient m)
        (final_value_fn.hessian m) in
      ctrl in
      
    (* Verification function *)
    let verification m =
      let test_fns = Viscosity.create_test_functions m in
      Array.for_all (fun test_fn ->
        Viscosity.verify_subsolution final_value_fn test_fn m &&
        Viscosity.verify_supersolution final_value_fn test_fn m
      ) test_fns in
      
    {
      value_function = final_value_fn;
      optimal_control;
      verification;
    }
end

(* Dynamic programming *)
module DP = struct  
  (* Value function type *)
  type value_function = {
    value: t -> float;
    gradient: t -> Tensor.t;
    hessian: t -> Tensor.t Tensor.t;
  }
  
  (* Dynamic programming operator *)
  module Operator = struct
    (* T operator from dynamic programming principle *)
    let apply_t_operator value_fn measure control dt =
      (* Compute immediate cost *)
      let cost = Cost.evaluate measure control.Control.sequence.coefficients in
      
      (* Compute controlled evolution *)
      let evolved_state = MVM.Evolution.evolve
        {MVM.measure; 
         filtration = Stochastic.create_complete_filtration [|0.0; dt|];
         time = 0.0;
         path = [|measure|]} 
        control (Tensor.randn [1]) dt in
        
      (* Compute continuation value *)
      let continuation = value_fn.value evolved_state.measure in
      
      Tensor.item cost +. exp (-.beta *. dt) *. continuation
      
    (* Optimal T operator *)
    let apply_optimal_t value_fn measure dt =
      let best_value = ref Float.infinity in
      let best_control = ref None in
      
      (* Sample controls *)
      for _ = 1 to 100 do
        let coeffs = Tensor.randn [10] in
        let r_vals = Tensor.ones [10] in
        let seq_opt = Control.create_sequence coeffs r_vals 10 in
        match seq_opt with
        | Some seq ->
            let ctrl_opt = Control.create_control 
              seq k (-.r, r) in
            begin match ctrl_opt with
            | Some ctrl ->
                let value = apply_t_operator value_fn measure ctrl dt in
                if value < !best_value then begin
                  best_value := value;
                  best_control := Some ctrl
                end
            | None -> ()
            end
        | None -> ()
      done;
      (!best_value, !best_control)
  end
  
  (* Value iteration *)
  module ValueIteration = struct
    (* Single iteration step *)
    let iteration_step value_fn measure dt =
      let (value, control) = Operator.apply_optimal_t value_fn measure dt in
      
      (* Update gradient *)
      let new_gradient = Derivatives.first_variation measure (fun m -> value) in
      
      (* Update hessian *)
      let new_hessian = Derivatives.second_variation measure (fun m -> value) in
      
      ({
        value = (fun m -> value);
        gradient = (fun m -> new_gradient);
        hessian = (fun m -> new_hessian);
      }, control)
      
    (* Full value iteration *)
    let iterate initial_value_fn measure max_iter dt tol =
      let rec iterate_aux value_fn iter =
        if iter >= max_iter then
          value_fn
        else
          let (new_value_fn, _) = iteration_step value_fn measure dt in
          
          (* Check convergence *)
          let diff = abs_float (value_fn.value measure -. new_value_fn.value measure) in
          if diff < tol then
            new_value_fn
          else
            iterate_aux new_value_fn (iter + 1)
      in
      iterate_aux initial_value_fn 0
  end
  
  (* Policy iteration *)
  module PolicyIteration = struct
    (* Policy evaluation step *)
    let evaluate_policy policy value_fn measure dt =
      Operator.apply_t_operator value_fn measure policy dt
      
    (* Policy improvement step *)
    let improve_policy value_fn measure dt =
      let (_, best_control) = Operator.apply_optimal_t value_fn measure dt in
      best_control
      
    (* Full policy iteration *)
    let iterate initial_policy initial_value_fn measure max_iter dt tol =
      let rec iterate_aux policy value_fn iter =
        if iter >= max_iter then
          (policy, value_fn)
        else
          (* Evaluation *)
          let value = evaluate_policy policy value_fn measure dt in
          let new_value_fn = {
            value = (fun m -> value);
            gradient = Derivatives.first_variation measure (fun m -> value);
            hessian = Derivatives.second_variation measure (fun m -> value);
          } in
          
          (* Improvement *)
          let new_policy_opt = improve_policy new_value_fn measure dt in
          
          match new_policy_opt with
          | Some new_policy ->
              (* Check convergence *)
              let diff = abs_float (value_fn.value measure -. new_value_fn.value measure) in
              if diff < tol then
                (new_policy, new_value_fn)
              else
                iterate_aux new_policy new_value_fn (iter + 1)
          | None -> (policy, value_fn)
      in
      iterate_aux initial_policy initial_value_fn 0
  end
end

(* Convergence analysis and divergence estimates *)
module ConvergenceAnalysis = struct  
  (* Divergence metrics *)
  type divergence_metrics = {
    value_divergence: float;
    gradient_divergence: float;
    control_divergence: float;
    total_variation: float;
    wasserstein: float;
  }
  
  (* Analysis of value function convergence *)
  module ValueConvergence = struct
    (* Compute divergence between two value functions *)
    let compute_value_divergence vf1 vf2 measure =
      let value1 = vf1.DP.value measure in
      let value2 = vf2.DP.value measure in
      let grad1 = vf1.DP.gradient measure in
      let grad2 = vf2.DP.gradient measure in
      
      {
        value_divergence = abs_float (value1 -. value2);
        gradient_divergence = Tensor.norm 
          (Tensor.sub grad1 grad2) 2.0 |> 
          Tensor.item;
        control_divergence = 0.0;
        total_variation = 0.0;
        wasserstein = 0.0;
      }
      
    (* Check Cauchy property for value function sequence *)
    let check_cauchy_sequence value_functions measure =
      let n = Array.length value_functions in
      let divergences = Array.init (n-1) (fun i ->
        compute_value_divergence 
          value_functions.(i) 
          value_functions.(i+1)
          measure
      ) in
      
      let is_cauchy = ref true in
      for i = 0 to n-3 do
        if divergences.(i).value_divergence > divergences.(i+1).value_divergence *. 2.0 then
          is_cauchy := false
      done;
      (!is_cauchy, divergences)
  end
  
  (* Divergence propagation analysis *)
  module DivergencePropagation = struct
    (* Analyze divergence propagation in value iteration *)
    let analyze_value_iteration value_fn measure controls dt steps =
      let divergences = Array.make steps {
        value_divergence = 0.0;
        gradient_divergence = 0.0;
        control_divergence = 0.0;
        total_variation = 0.0;
        wasserstein = 0.0;
      } in
      
      let state = ref {
        MVM.measure;
        filtration = Stochastic.create_complete_filtration [|0.0; dt|];
        time = 0.0;
        path = [|measure|];
      } in
      
      for i = 0 to steps - 1 do
        let ctrl = controls.(i mod (Array.length controls)) in
        let evolved = MVM.Evolution.evolve 
          !state ctrl (Tensor.randn [1]) dt in
        
        divergences.(i) <- {
          value_divergence = abs_float (
            value_fn.DP.value evolved.measure -.
            value_fn.DP.value !state.measure);
          gradient_divergence = Tensor.norm (
            Tensor.sub
              (value_fn.DP.gradient evolved.measure)
              (value_fn.DP.gradient !state.measure)) 2.0
            |> Tensor.item;
          control_divergence = 0.0;
          total_variation = Topology.total_variation 
            evolved.measure !state.measure;
          wasserstein = Topology.wasserstein_distance 
            evolved.measure !state.measure;
        };
        
        state := evolved
      done;
      divergences
      
    (* Compute stability bounds *)
    let compute_stability_bounds divergences =
      let value_bound = Array.fold_left 
        (fun acc e -> max acc e.value_divergence) 0.0 divergences in
      let gradient_bound = Array.fold_left 
        (fun acc e -> max acc e.gradient_divergence) 0.0 divergences in
      let measure_bound = Array.fold_left 
        (fun acc e -> max acc e.wasserstein) 0.0 divergences in
      (value_bound, gradient_bound, measure_bound)
  end
  
  (* Measure convergence rate analysis *)
  module MeasureConvergence = struct
    (* Compute convergence rate between measures *)
    let compute_rate measure1 measure2 dt =
      let dist = Topology.wasserstein_distance measure1 measure2 in
      dist /. dt
      
    (* Analyze convergence rates over sequence *)
    let analyze_sequence measures dt =
      let n = Array.length measures - 1 in
      let rates = Array.init n (fun i ->
        compute_rate measures.(i) measures.(i+1) dt
      ) in
      let mean_rate = Array.fold_left (+.) 0.0 rates /. float_of_int n in
      (mean_rate, rates)
  end
end

(* Cost function implementation *)
module Cost = struct
  open Types
  
  (* Cost function type *)
  type cost_function = {
    running_cost: Measure.t -> Control.control -> float;
    terminal_cost: Measure.t -> float option;
  }
  
  (* Create bounded cost function *)
  let create_bounded_cost bound =
    {
      running_cost = (fun measure control ->
        let h = Control.Evaluation.evaluate_h control in
        let value = Tensor.dot measure.density 
          (h (Tensor.float_tensor [|0.0|])) in
        min (Tensor.item value) bound
      );
      terminal_cost = None;
    }
    
  (* Evaluate cost function *)
  let evaluate measure control =
    let density = measure.Measure.density in
    let h = Control.Evaluation.evaluate_h control in
    Tensor.dot density (h (Tensor.float_tensor [|0.0|]))
    
  (* Compute discounted cost *)
  let compute_discounted_cost cost_fn state control time =
    match cost_fn.terminal_cost with
    | Some term_cost ->
        let running = cost_fn.running_cost state.MVM.measure control in
        let terminal = term_cost state.MVM.measure in
        running *. exp (-.beta *. time) +. terminal
    | None ->
        let running = cost_fn.running_cost state.MVM.measure control in
        running *. exp (-.beta *. time)
end