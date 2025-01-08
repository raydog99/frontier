open Torch

type time = float
type value = float array
type activity = float array

type process_state = {
  time: time;
  values: value;
  activity: activity;
}

let sum arr = Array.fold_left (+.) 0. arr

let brownian_increment dt =
  let std = sqrt dt in
  Tensor.randn [1] |> Tensor.mul_scalar std |> Tensor.to_float1

module type Process = sig
  type t
  val create : activity -> t
  val step : t -> float -> t
  val get_state : t -> process_state
end

module GLP = struct
  type t = {
    state: process_state;
    master_process: value;
  }

  let create activity =
    {
      state = {
        time = 0.0;
        values = Array.make (Array.length activity) 0.0;
        activity;
      };
      master_process = Array.make (Array.length activity) 0.0;
    }

  let from_levy_bridge activity horizon =
    let n = Array.length activity in
    {
      state = {
        time = 0.0;
        values = Array.make n 0.0;
        activity;
      };
      master_process = Array.make n horizon;
    }

  let compute_increments process dt =
    let t = process.state.time in
    let remaining = 1.0 -. t in
    let total_activity = sum process.state.activity in
    
    Array.mapi (fun i m ->
      let dw = brownian_increment dt in
      let drift = m /. total_activity *. dt in
      let diffusion = sqrt (m *. dt /. total_activity) *. dw.(0) in
      drift +. diffusion
    ) process.state.activity

  let update process dt increments =
    let new_time = process.state.time +. dt in
    let new_values = Array.map2 (+.) process.state.values increments in
    { process with
      state = {
        process.state with
        time = new_time;
        values = new_values;
      }
    }

  let step process dt =
    let increments = compute_increments process dt in
    update process dt increments

  let get_state process = process.state

  let get_terminal_distribution process =
    let n = Array.length process.state.activity in
    let total_activity = sum process.state.activity in
    Array.init n (fun i ->
      process.state.activity.(i) /. total_activity
    )
end

(* Lévy Bridge *)
module LevyBridge = struct
  type t = {
    process: process_state;
    terminal_law: float -> float;
    time_horizon: float;
  }

  let create terminal_law horizon =
    {
      process = {
        time = 0.0;
        values = [|0.0|];
        activity = [|1.0|];
      };
      terminal_law;
      time_horizon = horizon;
    }

  let transition_density bridge x y t s =
    let h_t = bridge.terminal_law in
    let remaining = bridge.time_horizon -. t in
    (h_t y) /. (h_t x) *. 
    exp (-. (y -. x) ** 2. /. (2. *. remaining))

  let get_state bridge = bridge.process
end

(* BLP *)
module BLP = struct
  type t = {
    base: process_state;
    sigma: float;
    bridges: float array;
  }

  let create activity =
    {
      base = {
        time = 0.0;
        values = Array.make (Array.length activity) 0.0;
        activity;
      };
      sigma = 1.0;
      bridges = Array.make (Array.length activity) 0.0;
    }

  let create_with_sigma activity sigma =
    { (create activity) with sigma }

  let step process dt =
    let t = process.base.time in
    if t >= 1.0 then process
    else
      let dim = Array.length process.base.values in
      let total_activity = sum process.base.activity in
      
      (* Update bridges *)
      let new_bridges = Array.mapi (fun i b ->
        let dw = brownian_increment dt in
        b +. sqrt (process.base.activity.(i)) *. dw.(0)
      ) process.bridges in
      
      (* Compute new values *)
      let new_values = Array.mapi (fun i v ->
        let m_i = process.base.activity.(i) in
        let z = brownian_increment dt in
        v +. t *. (m_i /. total_activity) *. process.sigma *. z.(0) +.
        process.sigma *. new_bridges.(i)
      ) process.base.values in
      
      { process with
        base = {
          process.base with
          time = t +. dt;
          values = new_values;
        };
        bridges = new_bridges;
      }

  let get_state process = process.base
  let get_bridges process = process.bridges

  let compute_covariance process =
    let dim = Array.length process.base.values in
    let cov = Array.make_matrix dim dim 0. in
    let total_activity = sum process.base.activity in
    
    for i = 0 to dim - 1 do
      for j = 0 to dim - 1 do
        cov.(i).(j) <- 
          process.base.activity.(i) *. process.base.activity.(j) /.
          total_activity *. process.sigma ** 2.
      done
    done;
    cov
end

module PLP = struct
  type t = {
    base: process_state;
    intensity: float;
    count: int array;
  }

  let create activity =
    {
      base = {
        time = 0.0;
        values = Array.make (Array.length activity) 0.0;
        activity;
      };
      intensity = 1.0;
      count = Array.make (Array.length activity) 0;
    }

  let create_with_intensity activity intensity =
    { (create activity) with intensity }

  let compute_intensity process =
    let t = process.base.time in
    if t >= 1.0 then 0.0
    else
      let total_count = Array.fold_left (+) 0 process.count in
      (process.intensity -. float_of_int total_count) /. (1. -. t)

  let generate_jumps process dt =
    let base_intensity = compute_intensity process in
    let total_activity = sum process.base.activity in
    let jumps = Array.make (Array.length process.base.values) 0 in
    
    if Random.float 1.0 < base_intensity *. dt then
      let v = Random.float total_activity in
      let rec assign_jump acc i =
        if i >= Array.length jumps then ()
        else
          let new_acc = acc +. process.base.activity.(i) in
          if v <= new_acc then
            jumps.(i) <- 1
          else
            assign_jump new_acc (i + 1)
      in
      assign_jump 0.0 0;
    jumps

  let step process dt =
    let jumps = generate_jumps process dt in
    let new_values = Array.map2 (+.) process.base.values 
      (Array.map float_of_int jumps) in
    let new_count = Array.map2 (+) process.count jumps in
    
    { process with
      base = {
        process.base with
        time = process.base.time +. dt;
        values = new_values;
      };
      count = new_count;
    }

  let get_state process = process.base
end

module MeasureChange = struct
  type density_process = {
    time: float;
    values: float array;
    radon_nikodym: float;
  }

  let create_density_process state =
    let time = state.time in
    let values = state.values in
    let r_t = sum values in
    let theta_t = if time >= 1.0 then 0.0
                 else 1. /. (1. -. time) in
    
    {
      time;
      values;
      radon_nikodym = theta_t *. r_t;
    }

  let compute_radon_nikodym density =
    density.radon_nikodym

  let change_measure state =
    let density = create_density_process state in
    {
      time = state.time;
      values = Array.map (fun v -> 
        v *. density.radon_nikodym
      ) state.values;
      activity = state.activity;
    }
end

module PathProperties = struct
  type path = {
    times: float array;
    values: float array array;
  }

  let compute_holder_exponent path =
    let values = path.values in
    let n = Array.length values in
    let dim = Array.length values.(0) in
    let max_scale = int_of_float (log10 (float_of_int n)) in
    
    Array.init dim (fun d ->
      let variations = Array.init max_scale (fun scale ->
        let step = 1 lsl scale in
        let var_sum = ref 0. in
        for i = 0 to n - step - 1 do
          let diff = abs_float (values.(i + step).(d) -. values.(i).(d)) in
          var_sum := !var_sum +. diff
        done;
        log (!var_sum /. float_of_int (n - step))
      ) in
      
      let x = Array.init max_scale (fun i -> 
        log (float_of_int (1 lsl i))) in
      let y = variations in
      
      (* Linear regression *)
      let sum_x = Array.fold_left (+.) 0. x in
      let sum_y = Array.fold_left (+.) 0. y in
      let sum_xy = Array.fold_left2 (fun acc xi yi -> 
        acc +. xi *. yi) 0. x y in
      let sum_xx = Array.fold_left (fun acc xi -> 
        acc +. xi *. xi) 0. x in
      let n_float = float_of_int (Array.length x) in
      
      (n_float *. sum_xy -. sum_x *. sum_y) /. 
      (n_float *. sum_xx -. sum_x *. sum_x)
    )

  let detect_jumps path threshold =
    let values = path.values in
    let times = path.times in
    let n = Array.length times in
    
    let rec find_jumps acc i =
      if i >= n - 1 then List.rev acc
      else
        let diffs = Array.map2 (fun x y -> 
          abs_float (y -. x)
        ) values.(i) values.(i+1) in
        
        if Array.exists (fun d -> d > threshold) diffs then
          find_jumps ((times.(i), Array.copy diffs) :: acc) (i + 1)
        else
          find_jumps acc (i + 1)
    in
    
    find_jumps [] 0

  let compute_occupation_density path n_bins =
    let values = path.values in
    let dim = Array.length values.(0) in
    
    let min_vals = Array.make dim infinity in
    let max_vals = Array.make dim neg_infinity in
    Array.iter (fun v ->
      Array.iteri (fun i x ->
        min_vals.(i) <- min min_vals.(i) x;
        max_vals.(i) <- max max_vals.(i) x
      ) v
    ) values;
    
    let density = Array.make_matrix dim n_bins 0. in
    Array.iter (fun v ->
      Array.iteri (fun i x ->
        let bin = int_of_float (
          float_of_int n_bins *. (x -. min_vals.(i)) /. 
          (max_vals.(i) -. min_vals.(i))
        ) in
        let bin_idx = min (n_bins - 1) (max 0 bin) in
        density.(i).(bin_idx) <- density.(i).(bin_idx) +. 1.
      ) v
    ) values;
    
    Array.map (fun row ->
      let total = Array.fold_left (+.) 0. row in
      Array.map (fun x -> x /. total) row
    ) density
end

module Numerical = struct
  type scheme = Euler | Milstein | RK4

  let euler_step state dt =
    let values = state.values in
    let activity = state.activity in
    let total_activity = sum activity in
    
    Array.mapi (fun i v ->
      let dw = brownian_increment dt in
      let drift = v *. dt *. activity.(i) /. total_activity in
      let diffusion = sqrt (activity.(i) *. dt) *. dw.(0) in
      drift +. diffusion
    ) values

  let milstein_step state dt =
    let values = state.values in
    let activity = state.activity in
    
    Array.mapi (fun i v ->
      let dw = brownian_increment dt in
      let drift = v *. dt *. activity.(i) in
      let diffusion = sqrt (activity.(i) *. dt) *. dw.(0) in
      let correction = 0.5 *. activity.(i) *. 
                      (dw.(0) *. dw.(0) -. dt) in
      drift +. diffusion +. correction
    ) values

  let rk4_step state dt =
    let f t v =
      Array.map2 (fun x a -> x *. a) v state.activity in
    
    let values = state.values in
    let k1 = f state.time values in
    let k2 = f (state.time +. 0.5 *. dt)
      (Array.map2 (fun v k -> v +. 0.5 *. dt *. k) values k1) in
    let k3 = f (state.time +. 0.5 *. dt)
      (Array.map2 (fun v k -> v +. 0.5 *. dt *. k) values k2) in
    let k4 = f (state.time +. dt)
      (Array.map2 (fun v k -> v +. dt *. k) values k3) in
    
    Array.init (Array.length values) (fun i ->
      dt /. 6. *. (k1.(i) +. 2. *. k2.(i) +. 2. *. k3.(i) +. k4.(i))
    )

  let solve state scheme dt steps =
    let rec iterate current_state acc remaining =
      if remaining <= 0 then List.rev acc
      else
        let increment = match scheme with
          | Euler -> euler_step current_state dt
          | Milstein -> milstein_step current_state dt
          | RK4 -> rk4_step current_state dt in
        
        let new_values = Array.map2 (+.) current_state.values increment in
        let new_state = {
          current_state with
          time = current_state.time +. dt;
          values = new_values;
        } in
        
        iterate new_state 
          ((new_state.time, Array.copy new_values) :: acc) 
          (remaining - 1)
    in
    
    iterate state [(state.time, Array.copy state.values)] steps

  let adaptive_step state error_tol =
    let initial_dt = 0.01 in
    let dt = ref initial_dt in
    let error = ref infinity in
    
    while !error > error_tol do
      let step1 = euler_step state !dt in
      let step2 = euler_step 
        {state with values = Array.map2 (+.) state.values step1} 
        !dt in
      let half_step = euler_step state (!dt /. 2.) in
      
      error := Array.fold_left2 (fun acc x y ->
        max acc (abs_float (x -. y))
      ) 0. step2 half_step;
      
      if !error > error_tol then
        dt := !dt /. 2.
    done;
    
    !dt, euler_step state !dt
end

module Simulation = struct
  type config = {
    n_paths: int;
    n_steps: int;
    dt: float;
    scheme: Numerical.scheme;
    antithetic: bool;
    stratification: int option;
  }

  module QuasiMonteCarlo = struct
    let sobol dim n =
      let direction_numbers = Array.make_matrix dim 32 0 in
      for d = 0 to dim - 1 do
        for i = 0 to 31 do
          direction_numbers.(d).(i) <- 1 lsl (31 - i)
        done
      done;
      
      let result = Array.make_matrix n dim 0. in
      for i = 0 to n - 1 do
        for d = 0 to dim - 1 do
          let mut = ref i in
          let value = ref 0 in
          for j = 0 to 31 do
            if !mut land (1 lsl j) <> 0 then
              value := !value lxor direction_numbers.(d).(j)
          done;
          result.(i).(d) <- float_of_int !value /. 2.0 ** 32.
        done
      done;
      result
  end

  module MultiLevel = struct
    type level = {
      dt: float;
      samples: float array array;
      correction: float array array;
    }

    let simulate process config n_levels =
      let base_dt = config.dt in
      Array.init n_levels (fun l ->
        let dt = base_dt /. (2. ** float_of_int l) in
        let coarse = Numerical.solve process.GLP.state config.scheme dt
          (config.n_steps / (1 lsl l)) in
        let fine = Numerical.solve process.GLP.state config.scheme
          (dt /. 2.) (config.n_steps / (1 lsl (l+1))) in
        
        let coarse_values = Array.of_list (List.map snd coarse) in
        let fine_values = Array.of_list (List.map snd fine) in
        
        {
          dt;
          samples = coarse_values;
          correction = Array.map2 (Array.map2 (-.)) 
            fine_values coarse_values;
        }
      )
  end

  let simulate_paths process config =
    let n_paths = if config.antithetic then config.n_paths / 2 
                 else config.n_paths in
    
    let base_paths = match config.stratification with
      | Some n_strata ->
          let n_per_stratum = n_paths / n_strata in
          let dim = Array.length process.GLP.state.activity in
          let stratified_samples = Array.init n_strata (fun i ->
            Array.init n_per_stratum (fun _ ->
              let u = float_of_int i +. Random.float 1.0 in
              let z = Tensor.erfinv (Tensor.of_float0 (2. *. u -. 1.)) 
                     |> Tensor.to_float0 in
              Array.make dim (z *. sqrt 2.)
            )
          ) |> Array.concat in
          
          Array.map (fun initial_values ->
            let path = Numerical.solve 
              {process.GLP.state with values = initial_values}
              config.scheme config.dt config.n_steps in
            {
              PathProperties.times = Array.of_list (List.map fst path);
              values = Array.of_list (List.map snd path);
            }
          ) stratified_samples
          
      | None ->
          Array.init n_paths (fun _ ->
            let path = Numerical.solve process.GLP.state
              config.scheme config.dt config.n_steps in
            {
              PathProperties.times = Array.of_list (List.map fst path);
              values = Array.of_list (List.map snd path);
            }
          )
    in
    
    if config.antithetic then
      Array.append base_paths
        (Array.map (fun path ->
          {path with values = Array.map (Array.map (~-.)) path.values}
        ) base_paths)
    else
      base_paths
end

module RiskMeasures = struct
  type risk_measure = {
    var: float array;
    cvar: float array;
    expected_shortfall: float array;
    maximum_drawdown: float array;
  }

  let compute_var values alpha =
    let dim = Array.length values.(0) in
    Array.init dim (fun d ->
      let sorted = Array.map (fun v -> v.(d)) values |> Array.copy in
      Array.sort compare sorted;
      let idx = int_of_float (float_of_int (Array.length sorted) *. alpha) in
      sorted.(idx)
    )

  let compute_cvar values alpha =
    let dim = Array.length values.(0) in
    let var = compute_var values alpha in
    Array.init dim (fun d ->
      let losses = Array.map (fun v -> v.(d)) values
                  |> Array.filter (fun x -> x <= var.(d)) in
      Array.fold_left (+.) 0. losses /. float_of_int (Array.length losses)
    )

  let compute_risk_measures values alpha =
    let var = compute_var values alpha in
    let cvar = compute_cvar values alpha in
    let dim = Array.length values.(0) in
    
    let max_drawdown = Array.init dim (fun d ->
      let series = Array.map (fun v -> v.(d)) values in
      let n = Array.length series in
      let running_max = ref series.(0) in
      let max_drawdown = ref 0. in
      
      for i = 1 to n - 1 do
        running_max := max !running_max series.(i);
        let drawdown = !running_max -. series.(i) in
        max_drawdown := max !max_drawdown drawdown
      done;
      !max_drawdown
    ) in
    
    let expected_shortfall = Array.init dim (fun d ->
      let sorted = Array.map (fun v -> v.(d)) values |> Array.copy in
      Array.sort compare sorted;
      let cutoff = int_of_float (float_of_int (Array.length sorted) *. alpha) in
      let sum = ref 0. in
      for i = 0 to cutoff - 1 do
        sum := !sum +. sorted.(i)
      done;
      !sum /. float_of_int cutoff
    ) in
    
    { var; cvar; expected_shortfall; maximum_drawdown }
end

module Analysis = struct
  type dependency_measure = {
    correlation: float array array;
    rank_correlation: float array array;
    tail_dependence: float array array;
    copula_estimate: float array array array;
  }

  let compute_correlation values =
    let dim = Array.length values.(0) in
    let n = Array.length values in
    let means = Array.make dim 0. in
    
    (* Compute means *)
    Array.iter (fun v ->
      Array.iteri (fun i x ->
        means.(i) <- means.(i) +. x
      ) v
    ) values;
    Array.iteri (fun i m ->
      means.(i) <- m /. float_of_int n
    ) means;
    
    (* Compute correlation matrix *)
    let corr = Array.make_matrix dim dim 0. in
    for i = 0 to dim - 1 do
      for j = i to dim - 1 do
        let cov = ref 0. in
        for k = 0 to n - 1 do
          cov := !cov +. (values.(k).(i) -. means.(i)) *.
                        (values.(k).(j) -. means.(j))
        done;
        let correlation = !cov /. sqrt (
          Array.fold_left (fun acc v ->
            acc +. (v.(i) -. means.(i)) ** 2.
          ) 0. values *.
          Array.fold_left (fun acc v ->
            acc +. (v.(j) -. means.(j)) ** 2.
          ) 0. values
        ) in
        corr.(i).(j) <- correlation;
        if i <> j then corr.(j).(i) <- correlation
      done
    done;
    corr

  let estimate_copula values n_bins =
    let dim = Array.length values.(0) in
    let n = Array.length values in
    
    (* Compute ranks *)
    let ranks = Array.init dim (fun d ->
      let col = Array.map (fun v -> v.(d)) values in
      let sorted_idx = Array.init n (fun i -> i) in
      Array.sort (fun i j -> compare col.(i) col.(j)) sorted_idx;
      let ranks = Array.make n 0 in
      Array.iteri (fun i idx -> ranks.(idx) <- i) sorted_idx;
      Array.map float_of_int ranks
    ) in
    
    (* Estimate empirical copula *)
    let copula = Array.make_matrix n_bins n_bins 0. in
    for i = 0 to n - 1 do
      let bin_indices = Array.map (fun rank ->
        min (n_bins - 1) (int_of_float (rank.(i) *. 
          float_of_int n_bins /. float_of_int n))
      ) ranks in
      copula.(bin_indices.(0)).(bin_indices.(1)) <- 
        copula.(bin_indices.(0)).(bin_indices.(1)) +. 1.
    done;
    
    (* Normalize *)
    Array.map (fun row ->
      Array.map (fun x -> x /. float_of_int n) row
    ) copula |> Array.map (fun x -> [|x|])

  let analyze_dependencies values =
    let correlation = compute_correlation values in
    let ranks = Array.init (Array.length values) (fun i ->
      Array.map float_of_int (Array.make (Array.length values.(0)) i)
    ) in
    let rank_correlation = compute_correlation ranks in
    let copula = estimate_copula values 20 in
    let dim = Array.length values.(0) in
    
    let tail_dependence = Array.make_matrix dim dim 0. in
    for i = 0 to dim - 1 do
      for j = i + 1 to dim - 1 do
        let upper_tail = ref 0 in
        let threshold = int_of_float (0.95 *. float_of_int (Array.length values)) in
        Array.iter (fun v ->
          if v.(i) > values.(threshold).(i) && 
             v.(j) > values.(threshold).(j) then
            incr upper_tail
        ) values;
        let coef = float_of_int !upper_tail /. 
                  (0.05 *. float_of_int (Array.length values)) in
        tail_dependence.(i).(j) <- coef;
        tail_dependence.(j).(i) <- coef
      done
    done;
    
    { correlation; rank_correlation; tail_dependence; copula_estimate = copula }
end

module Calibration = struct
  type calibration_result = {
    parameters: float array;
    error: float;
    iterations: int;
    convergence: bool;
  }

  type optimization_method = [
    | `GradientDescent of {
        learning_rate: float;
        momentum: float;
        max_iter: int;
        tolerance: float;
      }
    | `SimulatedAnnealing of {
        temp_schedule: int -> float;
        max_iter: int;
      }
  ]

  let objective process params observations =
    let process = { process with
      GLP.state = {
        process.GLP.state with
        activity = params
      }
    } in
    
    let config = {
      Simulation.n_paths = 100;
      n_steps = 1000;
      dt = 0.01;
      scheme = Numerical.Euler;
      antithetic = true;
      stratification = None;
    } in
    
    let predictions = Simulation.simulate_paths process config in
    let n = Array.length predictions in
    let error = ref 0. in
    
    for i = 0 to n - 1 do
      Array.iter2 (fun pred obs ->
        Array.iter2 (fun p o ->
          error := !error +. (p -. o) ** 2.
        ) pred obs
      ) predictions.(i).values observations
    done;
    
    !error /. float_of_int n

  let gradient params process observations h =
    let dim = Array.length params in
    Array.init dim (fun i ->
      let params_plus = Array.copy params in
      let params_minus = Array.copy params in
      params_plus.(i) <- params_plus.(i) +. h;
      params_minus.(i) <- params_minus.(i) -. h;
      
      (objective process params_plus observations -.
       objective process params_minus observations) /. (2. *. h)
    )

  let calibrate process observations = function
    | `GradientDescent {learning_rate; momentum; max_iter; tolerance} ->
        let params = Array.copy process.GLP.state.activity in
        let velocity = Array.make (Array.length params) 0. in
        let rec iterate iter prev_error =
          if iter >= max_iter then
            {
              parameters = params;
              error = prev_error;
              iterations = iter;
              convergence = false;
            }
          else
            let grad = gradient params process observations 1e-6 in
            let new_error = objective process params observations in
            
            if abs_float (new_error -. prev_error) < tolerance then
              {
                parameters = params;
                error = new_error;
                iterations = iter;
                convergence = true;
              }
            else begin
              Array.iteri (fun i g ->
                velocity.(i) <- momentum *. velocity.(i) -. 
                              learning_rate *. g;
                params.(i) <- params.(i) +. velocity.(i)
              ) grad;
              iterate (iter + 1) new_error
            end
        in
        iterate 0 (objective process params observations)

    | `SimulatedAnnealing {temp_schedule; max_iter} ->
        let params = Array.copy process.GLP.state.activity in
        let best_params = Array.copy params in
        let current_error = objective process params observations in
        let best_error = current_error in
        
        let rec iterate iter best_params best_error params current_error =
          if iter >= max_iter then
            {
              parameters = best_params;
              error = best_error;
              iterations = iter;
              convergence = true;
            }
          else
            let temp = temp_schedule iter in
            let neighbor = Array.map (fun p ->
              p +. (Random.float 2. -. 1.) *. temp
            ) params in
            
            let neighbor_error = objective process neighbor observations in
            let accept = neighbor_error < current_error ||
              Random.float 1. < exp ((current_error -. neighbor_error) /. temp)
            in
            
            let (next_params, next_error) =
              if accept then (neighbor, neighbor_error)
              else (params, current_error)
            in
            
            let (new_best_params, new_best_error) =
              if next_error < best_error then
                (Array.copy next_params, next_error)
              else
                (best_params, best_error)
            in
            
            iterate (iter + 1) new_best_params new_best_error 
                   next_params next_error
        in
        
        iterate 0 best_params best_error params current_error
end

module StochasticIntegral = struct
  type decomposition = {
    drift: float array;
    martingale: float array;
    compensator: float array;
  }

  let compute_decomposition state =
    let time = state.time in
    let values = state.values in
    let activity = state.activity in
    
    if time >= 1.0 then
      {
        drift = Array.make (Array.length values) 0.;
        martingale = Array.make (Array.length values) 0.;
        compensator = Array.make (Array.length values) 0.;
      }
    else
      let remaining = 1. -. time in
      let drift = Array.map2 (fun v a ->
        v *. a /. remaining
      ) values activity in
      
      let martingale = Array.map2 (fun d v ->
        v -. time *. d
      ) drift values in
      
      let compensator = Array.map2 (fun m v ->
        v -. m
      ) martingale values in
      
      { drift; martingale; compensator }

  let verify_harness state (t1, t2, t3, t4) =
    let values = state.values in
    
    if t1 >= t2 || t2 >= t3 || t3 >= t4 then false
    else
      let h_tu = (t3 -. t2) /. (t4 -. t1) in
      let interpolated = Array.map (fun v ->
        v *. h_tu
      ) values in
      
      Array.for_all2 (fun actual expected ->
        abs_float (actual -. expected) < 1e-10
      ) values interpolated
end

let check_weak state history i =
  abs_float (history.(i) -. state.values.(i)) < 1e-10

let check_strong state history i =
  let check_independence =
    Array.fold_lefti (fun acc j v ->
      if j = i then acc
      else acc && abs_float v < 1e-10
    ) true state.values
  in
  check_weak state history i && check_independence