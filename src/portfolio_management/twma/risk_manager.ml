open Torch

type t = {
  max_drawdown: float;
  var_threshold: float;
  es_threshold: float;
  volatility_target: float;
  max_leverage: float;
  correlation_threshold: float;
  mutable historical_returns: float array;
}

let create ?(max_drawdown=0.2) ?(var_threshold=0.05) ?(es_threshold=0.1)
           ?(volatility_target=0.15) ?(max_leverage=2.0) ?(correlation_threshold=0.7) () =
  {
    max_drawdown;
    var_threshold;
    es_threshold;
    volatility_target;
    max_leverage;
    correlation_threshold;
    historical_returns = [||];
  }

let update_historical_returns t new_returns =
  t.historical_returns <- Array.append t.historical_returns new_returns

let calculate_var t confidence_level =
  let sorted_returns = Array.copy t.historical_returns |> Array.sort compare in
  let index = int_of_float (float_of_int (Array.length sorted_returns) *. (1. -. confidence_level)) in
  sorted_returns.(index)

let calculate_es t confidence_level =
  let var = calculate_var t confidence_level in
  let tail_returns = Array.filter (fun r -> r <= var) t.historical_returns in
  Array.fold_left (+.) 0. tail_returns /. float_of_int (Array.length tail_returns)

let calculate_portfolio_var weights returns confidence_level =
  let portfolio_returns = 
    Array.map (fun day_returns ->
      Array.fold_left2 (fun acc w r -> acc +. w *. r) 0. weights day_returns
    ) returns
  in
  let sorted_returns = Array.copy portfolio_returns |> Array.sort compare in
  let index = int_of_float (float_of_int (Array.length sorted_returns) *. (1. -. confidence_level)) in
  sorted_returns.(index)

let adjust_weights_for_risk t weights returns =
  let num_assets = Array.length weights in
  let correlation_matrix = Tensor.corrcoef (Tensor.of_float2 returns) in
  
  let adjusted_weights = Array.mapi (fun i w ->
    let correlated_assets = ref 0 in
    for j = 0 to num_assets - 1 do
      if i <> j && Tensor.get correlation_matrix [i; j] |> Tensor.to_float0_exn > t.correlation_threshold then
        incr correlated_assets
    done;
    w /. (1. +. float_of_int !correlated_assets)
  ) weights in

  let portfolio_var = calculate_portfolio_var adjusted_weights returns t.var_threshold in
  let var_adjustment = min 1. (t.var_threshold /. abs_float portfolio_var) in
  let portfolio_es = calculate_es t t.es_threshold in
  let es_adjustment = min 1. (t.es_threshold /. abs_float portfolio_es) in

  let portfolio_volatility = 
    let mean_return = Array.fold_left (+.) 0. t.historical_returns /. float_of_int (Array.length t.historical_returns) in
    sqrt (Array.fold_left (fun acc r -> acc +. (r -. mean_return) ** 2.) 0. t.historical_returns 
          /. float_of_int (Array.length t.historical_returns))
  in
  let volatility_adjustment = t.volatility_target /. portfolio_volatility in

  let final_weights = Array.map (fun w -> 
    w *. var_adjustment *. es_adjustment *. volatility_adjustment
  ) adjusted_weights in

  let total_exposure = Array.fold_left (+.) 0. final_weights in
  if total_exposure > t.max_leverage then
    Array.map (fun w -> w *. t.max_leverage /. total_exposure) final_weights
  else
    final_weights

let apply_risk_management t weights returns =
  let adjusted_weights = adjust_weights_for_risk t weights returns in
  update_historical_returns t (Array.map (Array.fold_left2 (fun acc w r -> acc +. w *. r) 0. adjusted_weights) returns);
  adjusted_weights 