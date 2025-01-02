open Torch

type t = {
  n_assets: int;
  mutable returns: Tensor.t;
  covariance: Tensor.t;
}

type strategy = SingleStage | TwoStage

let create n_assets returns covariance =
  let returns = Tensor.of_float_array returns [n_assets] in
  let covariance = Tensor.of_float_array2 covariance in
  { n_assets; returns; covariance }

let qubo_formulation t theta k target_return m =
  let n = t.n_assets in
  let q = Tensor.zeros [n * k; n * k] in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      for k1 = 0 to k - 1 do
        for k2 = 0 to k - 1 do
          let idx1 = i * k + k1 in
          let idx2 = j * k + k2 in
          let cov_term = Tensor.get t.covariance [i; j] *. (2. ** float_of_int (k1 + k2 - 2 * k + 2)) in
          Tensor.set q [idx1; idx2] (Tensor.get q [idx1; idx2] +. theta *. cov_term)
        done
      done
    done
  done;
  
  for i = 0 to n - 1 do
    for k1 = 0 to k - 1 do
      let idx = i * k + k1 in
      let return_term = Tensor.get t.returns [i] *. (2. ** float_of_int (k1 - k + 1)) in
      Tensor.set q [idx; idx] (Tensor.get q [idx; idx] -. m *. return_term)
    done
  done;
  
  q

let binary_to_weights binary_solution k =
  let n = Array.length binary_solution / k in
  Array.init n (fun i ->
    let sum = ref 0. in
    for j = 0 to k - 1 do
      sum := !sum +. (if binary_solution.(i * k + j) > 0.5 then 2. ** float_of_int (j - k + 1) else 0.)
    done;
    !sum
  )

let calculate_energy q solution =
  let energy = ref 0. in
  for i = 0 to Array.length solution - 1 do
    for j = 0 to Array.length solution - 1 do
      energy := !energy +. solution.(i) *. solution.(j) *. Tensor.get q [i; j]
    done
  done;
  !energy

let simulated_annealing q initial_temp final_temp cooling_rate max_iterations =
  let n = Tensor.shape q |> List.hd in
  let current_solution = Array.init n (fun _ -> if Random.bool () then 1. else 0.) in
  let best_solution = Array.copy current_solution in
  let current_energy = calculate_energy q current_solution in
  let best_energy = ref current_energy in
  
  let rec annealing temp iter =
    if temp > final_temp && iter < max_iterations then
      let i = Random.int n in
      let new_solution = Array.copy current_solution in
      new_solution.(i) <- 1. -. new_solution.(i);
      let new_energy = calculate_energy q new_solution in
      let delta_e = new_energy -. current_energy in
      
      if delta_e < 0. || Random.float 1. < exp (-. delta_e /. temp) then begin
        for j = 0 to n - 1 do current_solution.(j) <- new_solution.(j) done;
        if new_energy < !best_energy then begin
          for j = 0 to n - 1 do best_solution.(j) <- new_solution.(j) done;
          best_energy := new_energy;
        end;
        annealing (temp *. cooling_rate) (iter + 1)
      end else
        annealing (temp *. cooling_rate) (iter + 1)
    else
      best_solution
  in
  
  annealing initial_temp 0

let optimize t target_return k theta m max_iterations =
  let q = qubo_formulation t theta k target_return m in
  let binary_solution = simulated_annealing q 100. 0.1 0.995 max_iterations in
  binary_to_weights binary_solution k

let estimate_penalty_coefficient t target_return n_samples =
  let n = t.n_assets in
  let generate_random_solution () =
    Array.init n (fun _ -> Random.float 1.)
  in
  
  let solutions = Array.init n_samples (fun _ -> generate_random_solution ()) in
  
  let best_feasible = ref None in
  let worst_infeasible = ref None in
  
  Array.iter (fun sol ->
    let portfolio_return = Tensor.dot (Tensor.of_float_array sol [n]) t.returns |> Tensor.to_float0 in
    if portfolio_return >= target_return then
      match !best_feasible with
      | None -> best_feasible := Some sol
      | Some best ->
        if Tensor.dot (Tensor.of_float_array sol [n]) (Tensor.matmul t.covariance (Tensor.of_float_array sol [n; 1])) |> Tensor.to_float0
           < Tensor.dot (Tensor.of_float_array best [n]) (Tensor.matmul t.covariance (Tensor.of_float_array best [n; 1])) |> Tensor.to_float0
        then best_feasible := Some sol
    else
      match !worst_infeasible with
      | None -> worst_infeasible := Some sol
      | Some worst ->
        if portfolio_return > Tensor.dot (Tensor.of_float_array worst [n]) t.returns |> Tensor.to_float0
        then worst_infeasible := Some sol
  ) solutions;
  
  match !best_feasible, !worst_infeasible with
  | Some best, Some worst ->
    let best_risk = Tensor.dot (Tensor.of_float_array best [n]) (Tensor.matmul t.covariance (Tensor.of_float_array best [n; 1])) |> Tensor.to_float0 in
    let worst_risk = Tensor.dot (Tensor.of_float_array worst [n]) (Tensor.matmul t.covariance (Tensor.of_float_array worst [n; 1])) |> Tensor.to_float0 in
    let best_return = Tensor.dot (Tensor.of_float_array best [n]) t.returns |> Tensor.to_float0 in
    let worst_return = Tensor.dot (Tensor.of_float_array worst [n]) t.returns |> Tensor.to_float0 in
    (best_risk -. worst_risk) /. ((target_return -. worst_return) ** 2. -. (best_return -. target_return) ** 2.)
  | _ -> 1000.

let two_stage_search t target_return k theta m max_iterations =
  let q1 = qubo_formulation t theta k target_return m in
  let initial_solution = simulated_annealing q1 100. 0.1 0.995 (max_iterations / 2) in
  
  let refined_q = Tensor.clone q1 in
  for i = 0 to Array.length initial_solution - 1 do
    if initial_solution.(i) > 0.5 then
      Tensor.set refined_q [i; i] (Tensor.get refined_q [i; i] -. m)
  done;
  
  let final_solution = simulated_annealing refined_q 50. 0.01 0.99 (max_iterations / 2) in
  binary_to_weights final_solution k

let calculate_portfolio_return t weights =
  Tensor.dot (Tensor.of_float_array weights [t.n_assets]) t.returns
  |> Tensor.to_float0

let calculate_portfolio_risk t weights =
  let w = Tensor.of_float_array weights [t.n_assets; 1] in
  Tensor.matmul (Tensor.matmul (Tensor.transpose w 0 1) t.covariance) w
  |> Tensor.to_float0
  |> sqrt

let calculate_sharpe_ratio t weights risk_free_rate =
  let portfolio_return = calculate_portfolio_return t weights in
  let portfolio_risk = calculate_portfolio_risk t weights in
  (portfolio_return -. risk_free_rate) /. portfolio_risk

let backtest t strategy initial_target_return k theta m max_iterations n_quarters =
  let quarterly_returns = Array.make n_quarters 0. in
  let current_weights = ref (Array.make t.n_assets (1. /. float_of_int t.n_assets)) in
  let current_target_return = ref initial_target_return in

  for i = 0 to n_quarters - 1 do
    current_weights := (match strategy with
      | SingleStage -> optimize t !current_target_return k theta m max_iterations
      | TwoStage -> two_stage_search t !current_target_return k theta m max_iterations
    );

    let quarter_return = calculate_portfolio_return t !current_weights in
    quarterly_returns.(i) <- quarter_return;

    current_target_return := !current_target_return *. (1. +. quarter_return);

    let price_changes = Array.init t.n_assets (fun _ -> 1. +. (Random.float 0.2 -. 0.1)) in
    t.returns <- Tensor.mul t.returns (Tensor.of_float_array price_changes [t.n_assets]);
  done;

  quarterly_returns

let compare_strategies t initial_target_return k theta m max_iterations n_quarters =
  let single_stage_returns = backtest t SingleStage initial_target_return k theta m max_iterations n_quarters in
  let two_stage_returns = backtest t TwoStage initial_target_return k theta m max_iterations n_quarters in
  (single_stage_returns, two_stage_returns)

let parameter_sensitivity_analysis t target_return k_values theta_values m_values max_iterations =
  let results = ref [] in
  List.iter (fun k ->
    List.iter (fun theta ->
      List.iter (fun m ->
        let weights = two_stage_search t target_return k theta m max_iterations in
        let portfolio_return = calculate_portfolio_return t weights in
        let portfolio_risk = calculate_portfolio_risk t weights in
        let sharpe_ratio = calculate_sharpe_ratio t weights 0.02 in (* 2% risk-free rate *)
        results := (k, theta, m, sharpe_ratio) :: !results
      ) m_values
    ) theta_values
  ) k_values;
  !results