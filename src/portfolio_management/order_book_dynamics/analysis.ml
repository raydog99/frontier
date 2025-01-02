let calculate_realized_volatility price_series =
  let log_returns = 
    List.map2 (fun p1 p2 -> log (float_of_int p2 /. float_of_int p1)) 
      price_series (List.tl price_series) 
  in
  let squared_returns = List.map (fun r -> r *. r) log_returns in
  let sum_squared_returns = List.fold_left (+.) 0.0 squared_returns in
  sqrt (sum_squared_returns /. float_of_int (List.length squared_returns))

let linear_regression x y =
  let n = float_of_int (List.length x) in
  let sum_x = List.fold_left (+.) 0. x in
  let sum_y = List.fold_left (+.) 0. y in
  let sum_xy = List.fold_left2 (fun acc xi yi -> acc +. xi *. yi) 0. x y in
  let sum_xx = List.fold_left (fun acc xi -> acc +. xi *. xi) 0. x in
  let slope = (n *. sum_xy -. sum_x *. sum_y) /. (n *. sum_xx -. sum_x *. sum_x) in
  let intercept = (sum_y -. slope *. sum_x) /. n in
  (slope, intercept)

let calculate_sharpe_ratio returns risk_free_rate =
  let excess_returns = List.map (fun r -> r -. risk_free_rate) returns in
  let avg_excess_return = List.fold_left (+.) 0. excess_returns /. float_of_int (List.length excess_returns) in
  let variance = List.fold_left (fun acc r -> acc +. (r -. avg_excess_return) ** 2.) 0. excess_returns /. float_of_int (List.length excess_returns) in
  let std_dev = sqrt variance in
  avg_excess_return /. std_dev *. sqrt 252.  (* Annualized Sharpe ratio assuming daily returns *)

let calculate_sortino_ratio returns risk_free_rate =
  let excess_returns = List.map (fun r -> r -. risk_free_rate) returns in
  let avg_excess_return = List.fold_left (+.) 0. excess_returns /. float_of_int (List.length excess_returns) in
  let downside_returns = List.filter (fun r -> r < 0.) excess_returns in
  let downside_deviation = sqrt (List.fold_left (fun acc r -> acc +. r ** 2.) 0. downside_returns /. float_of_int (List.length downside_returns)) in
  avg_excess_return /. downside_deviation *. sqrt 252.  (* Annualized Sortino ratio assuming daily returns *)

let calculate_maximum_drawdown returns =
  let rec loop peak drawdown current = function
    | [] -> drawdown
    | r :: rest ->
        let new_peak = max peak r in
        let new_drawdown = max drawdown ((new_peak -. r) /. new_peak) in
        loop new_peak new_drawdown r rest
  in
  match returns with
  | [] -> 0.
  | hd :: tl -> loop hd 0. hd tl

let calculate_var returns confidence_level =
  let sorted_returns = List.sort compare returns in
  let index = int_of_float (float_of_int (List.length returns) *. (1. -. confidence_level)) in
  List.nth sorted_returns index

let calculate_expected_shortfall returns confidence_level =
  let var = calculate_var returns confidence_level in
  let tail_returns = List.filter (fun r -> r < var) returns in
  List.fold_left (+.) 0. tail_returns /. float_of_int (List.length tail_returns)