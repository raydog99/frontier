open Torch

type market_params = {
  kappa: float;
  num_stocks: int;
  time_horizon: float;
}

type numerical_params = {
  time_steps: int;
  space_steps: int;
  monte_carlo_paths: int;
  error_tolerance: float;
}

type solution = {
  stocks: Tensor.t;
  bessel: Tensor.t array;
  arbitrage: Tensor.t;
  time_changes: float array;
  error_metrics: error_metrics;
}

type error_metrics = {
  l2_error: float;
  max_error: float;
  relative_error: float;
  convergence_rate: float;
  confidence_intervals: (float * float) array;
}

type state = {
  stocks: Tensor.t;
  total_market: Tensor.t;
  time: float;
}

type market_params = {
  kappa: float;
  num_stocks: int;
  time_horizon: float;
}

type numerical_params = {
  time_steps: int;
  space_steps: int;
  monte_carlo_paths: int;
  error_tolerance: float;
}

let create_initial_state values =
  let device = Device.cuda_if_available () in
  let stocks = Tensor.of_float1 values ~device in
  let total_market = Tensor.sum stocks ~dim:[0] ~keepdim:true in
  { stocks; total_market; time = 0.0 }

let time_change state dt =
  let market_value = Tensor.float_value state.total_market in
  market_value /. 4.0 *. dt

let generate_brownian_increments ~num_steps ~num_stocks ~device dt =
  let dW = Tensor.randn [num_steps; num_stocks] ~device in
  Tensor.mul_scalar dW (sqrt dt)

let simulate_bessel_process ~dimension ~initial_value ~num_steps dt =
  let device = Device.cuda_if_available () in
  let process = Tensor.zeros [num_steps + 1] ~device in
  Tensor.copy_ (Tensor.narrow process ~dim:0 ~start:0 ~length:1)
    (Tensor.full [1] initial_value ~device);

  for t = 0 to num_steps - 1 do
    let current = Tensor.narrow process ~dim:0 ~start:t ~length:1 in
    let dW = Tensor.randn [1] ~device in
    let scaled_dW = Tensor.mul_scalar dW (sqrt dt) in

    let m = float dimension in
    let drift = Tensor.mul_scalar current ((m -. 1.0) /. 2.0) in
    let diffusion = Tensor.mul
      (Tensor.sqrt (Tensor.abs current))
      (Tensor.mul_scalar scaled_dW 2.0) in

    let next_value = Tensor.add
      (Tensor.add current (Tensor.mul_scalar drift dt))
      diffusion in

    Tensor.copy_
      (Tensor.narrow process ~dim:0 ~start:(t+1) ~length:1)
      next_value
  done;
  process

let simulate_market params num_params state =
  let device = state.stocks.device in
  let dt = params.time_horizon /. float num_params.time_steps in
  
  let stock_paths = Tensor.zeros 
    [num_params.time_steps + 1; params.num_stocks] ~device in
  
  Tensor.copy_ 
    (Tensor.narrow stock_paths ~dim:0 ~start:0 ~length:1)
    (Tensor.unsqueeze state.stocks ~dim:0);

  let dW = generate_brownian_increments 
    ~num_steps:num_params.time_steps
    ~num_stocks:params.num_stocks
    ~device dt in

  for t = 0 to num_params.time_steps - 1 do
    let current = Tensor.narrow stock_paths ~dim:0 ~start:t ~length:1 in
    let total = Tensor.sum current ~dim:[1] ~keepdim:true in
    
    let drift = Tensor.mul current (Tensor.full_like current params.kappa) in
    
    let vol = Tensor.mul 
      (Tensor.sqrt current)
      (Tensor.sqrt total) in
    let dW_t = Tensor.narrow dW ~dim:0 ~start:t ~length:1 in
    let diffusion = Tensor.mul vol dW_t in

    let next_value = Tensor.add
      (Tensor.add current (Tensor.mul_scalar drift dt))
      diffusion in

    Tensor.copy_
      (Tensor.narrow stock_paths ~dim:0 ~start:(t+1) ~length:1)
      next_value
  done;

  let final_stocks = Tensor.narrow stock_paths 
    ~dim:0 ~start:num_params.time_steps ~length:1 in
  let final_total = Tensor.sum final_stocks ~dim:[1] ~keepdim:true in

  { stocks = final_stocks;
    total_market = final_total;
    time = params.time_horizon }

let compute_optimal_arbitrage params num_params state =
  let device = state.stocks.device in
  let paths = Array.init num_params.monte_carlo_paths (fun _ ->
    let final_state = simulate_market params num_params state in
    final_state.stocks
  ) in
  
  let terminal_values = Tensor.stack (Array.to_list paths) ~dim:0 in
  let terminal_products = Tensor.prod terminal_values ~dim:[1] in
  let terminal_sums = Tensor.sum terminal_values ~dim:[1] in
  
  let current_product = Tensor.prod state.stocks ~dim:[0] in
  let current_sum = Tensor.sum state.stocks ~dim:[0] in
  
  Tensor.div
    (Tensor.mul current_product 
       (Tensor.mean (Tensor.div terminal_sums terminal_products)))
    current_sum

let compute_error_metrics ~numerical_sol ~bessel_paths ~arbitrage =
  let device = numerical_sol.device in
  
  let bessel_tensor = Tensor.stack (Array.to_list bessel_paths) ~dim:0 in
  let diff = Tensor.sub numerical_sol bessel_tensor in
  
  let l2_error = Tensor.norm diff ~p:2 |> Tensor.float_value in
  let max_error = Tensor.max diff |> Tensor.float_value in
  let relative_error = l2_error /. 
    (Tensor.norm numerical_sol ~p:2 |> Tensor.float_value) in
  
  let coarse = Tensor.norm numerical_sol ~p:2 |> Tensor.float_value in
  let fine = Tensor.norm bessel_tensor ~p:2 |> Tensor.float_value in
  let convergence_rate = log (abs_float (fine -. coarse)) /. log 2.0 in
  
  let n_bootstrap = 1000 in
  let bootstrap_samples = Array.init n_bootstrap (fun _ ->
    let idx = Tensor.randint ~high:(Tensor.size numerical_sol 0) 
                ~size:[Tensor.size numerical_sol 0] ~device in
    let sample = Tensor.index_select numerical_sol ~dim:0 ~index:idx in
    Tensor.mean sample |> Tensor.float_value
  ) in
  Array.sort compare bootstrap_samples;
  let confidence_intervals = 
    [|(bootstrap_samples.(25), bootstrap_samples.(975))|] in

  { l2_error;
    max_error;
    relative_error;
    convergence_rate;
    confidence_intervals }

let interpolate state1 state2 t =
  let alpha = t -. state1.time /. (state2.time -. state1.time) in
  let stocks = Tensor.add
    (Tensor.mul_scalar state1.stocks (1.0 -. alpha))
    (Tensor.mul_scalar state2.stocks alpha) in
  let total = Tensor.sum stocks ~dim:[0] ~keepdim:true in
  { stocks; total_market = total; time = t }

let validate_solution solution =
  let non_negative = Tensor.ge solution.stocks (Tensor.zeros_like solution.stocks) in
  let non_negative_valid = Tensor.all non_negative |> Tensor.bool_value in
  
  let convergence_valid = solution.error_metrics.convergence_rate > -1.0 in
  
  let intervals_valid = Array.for_all (fun (low, high) ->
    high > low && high -. low < 1.0
  ) solution.error_metrics.confidence_intervals in
  
  non_negative_valid && convergence_valid && intervals_valid

let solve market_params num_params initial_state =
  let final_state = simulate_market market_params num_params initial_state in
  
  let bessel_paths = Array.init market_params.num_stocks (fun i ->
    let initial_value = Tensor.float_value
      (Tensor.narrow initial_state.stocks ~dim:0 ~start:i ~length:1) in
    simulate_bessel_process
      ~dimension:(4 * int_of_float market_params.kappa)
      ~initial_value
      ~num_steps:num_params.time_steps
      (market_params.time_horizon /. float num_params.time_steps)
  ) in
  
  let arbitrage = compute_optimal_arbitrage 
    market_params num_params initial_state in
  
  let time_changes = Array.init num_params.time_steps (fun t ->
    let state_t = interpolate initial_state final_state 
      (float t *. market_params.time_horizon /. 
       float num_params.time_steps) in
    time_change state_t 
      (market_params.time_horizon /. float num_params.time_steps)
  ) in
  
  let error_metrics = compute_error_metrics
    ~numerical_sol:final_state.stocks
    ~bessel_paths
    ~arbitrage in
  
  { stocks = final_state.stocks;
    bessel = bessel_paths;
    arbitrage;
    time_changes;
    error_metrics }