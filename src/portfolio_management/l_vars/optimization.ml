open Torch

type t = {
    initial_holdings: float;
    liquidation_time: float;
    num_intervals: int;
    risk_aversion: float;
    volatility: float;
    permanent_impact: float;
    temporary_impact: float;
    drift: float;
  }

let create initial_holdings liquidation_time num_intervals risk_aversion volatility permanent_impact temporary_impact drift =
    { initial_holdings; liquidation_time; num_intervals; risk_aversion; volatility; permanent_impact; temporary_impact; drift }

let objective_function t trajectory =
    let { initial_holdings; liquidation_time; num_intervals; risk_aversion; volatility; permanent_impact; temporary_impact; drift } = t in
    let interval_length = liquidation_time /. float_of_int num_intervals in
    
    let trade_list = Tensor.(trajectory - (trajectory.roll ~shifts:1 ~dims:[0])) in
    let variance = Tensor.(sum (pow trajectory (f 2.)) * (f (volatility *. volatility *. interval_length))) in
    let permanent_cost = Tensor.(sum (trade_list * trade_list) * (f permanent_impact)) in
    let temporary_cost = Tensor.(sum (abs trade_list * (f temporary_impact))) in
    let expected_cost = Tensor.(permanent_cost + temporary_cost) in
    
    let drift_effect = Tensor.(sum (trajectory * (f (drift *. interval_length)))) in
    
    Tensor.(expected_cost + (f risk_aversion * variance) - drift_effect)

let optimize t initial_trajectory =
    let optimizer = Optimizer.adam [initial_trajectory] ~lr:0.01 in
    
    let rec optimize_loop iter best_loss best_trajectory =
      if iter >= 1000 then best_trajectory
      else
        let loss = objective_function t initial_trajectory in
        loss.backward();
        Optimizer.step optimizer;
        Optimizer.zero_grad optimizer;
        if Tensor.to_float0_exn loss < best_loss then
          optimize_loop (iter + 1) (Tensor.to_float0_exn loss) initial_trajectory
        else
          optimize_loop (iter + 1) best_loss best_trajectory
    in

    optimize_loop 0 Float.infinity initial_trajectory

let optimize_almgren_chriss t initial_trajectory =
    let { initial_holdings; liquidation_time; num_intervals; risk_aversion; volatility; temporary_impact; _ } = t in
    let interval_length = liquidation_time /. float_of_int num_intervals in
    
    let kappa = sqrt (risk_aversion *. volatility *. volatility /. temporary_impact) in
    let tau = float_of_int num_intervals *. interval_length in
    
    let optimal_trajectory = Tensor.zeros [num_intervals + 1] in
    for i = 0 to num_intervals do
      let t_i = float_of_int i *. interval_length in
      let x_t = initial_holdings *. (sinh (kappa *. (tau -. t_i)) /. sinh (kappa *. tau)) in
      Tensor.set optimal_trajectory i (Tensor.of_float0 x_t);
    done;
    
    optimal_trajectory

let optimize_dynamic t initial_trajectory price_dynamics =
    let rec optimize_step current_trajectory remaining_steps =
      if remaining_steps = 0 then current_trajectory
      else
        let current_price = price_dynamics current_trajectory in
        let updated_trajectory = optimize t current_trajectory in
        optimize_step updated_trajectory (remaining_steps - 1)
    in
    
    optimize_step initial_trajectory 10  (* Adjust the number of steps as needed *)

let optimize_constrained t initial_trajectory constraint_fn =
    let constrained_objective trajectory =
      let obj_value = objective_function t trajectory in
      let constraint_value = constraint_fn trajectory in
      Tensor.(obj_value + (f 1e6 * (max (constraint_value - f 0.) (f 0.))))
    in
    
    let optimizer = Optimizer.adam [initial_trajectory] ~lr:0.01 in
    
    let rec optimize_loop iter best_loss best_trajectory =
      if iter >= 1000 then best_trajectory
      else begin
        let loss = constrained_objective initial_trajectory in
        loss.backward();
        Optimizer.step optimizer;
        Optimizer.zero_grad optimizer;
        if Tensor.to_float0_exn loss < best_loss then
          optimize_loop (iter + 1) (Tensor.to_float0_exn loss) initial_trajectory
        else
          optimize_loop (iter + 1) best_loss best_trajectory
      end
    in
    
    optimize_loop 0 Float.infinity initial_trajectory