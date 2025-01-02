
type risk_measure =
  | ValueAtRisk of float
  | ConditionalVaR of float
  | ExpectedShortfall of float
  | MaxDrawdown

type risk_limit =
  | AbsoluteLimit of float
  | RelativeLimit of float

type t = {
  measure: risk_measure;
  limit: risk_limit;
}

type advanced_risk_measure =
  | TailValueAtRisk of float
  | ExpectedTail of float
  | ConditionalDrawdown of float

let calculate_risk portfolio risk_measure =
  let returns = Portfolio.get_returns portfolio in
  match risk_measure with
  | ValueAtRisk confidence ->
      let sorted_returns = List.sort compare returns in
      let index = int_of_float (float_of_int (List.length returns) *. (1. -. confidence)) in
      List.nth sorted_returns index
  | ConditionalVaR confidence ->
      let sorted_returns = List.sort compare returns in
      let cutoff_index = int_of_float (float_of_int (List.length returns) *. (1. -. confidence)) in
      let tail_returns = List.filteri (fun i _ -> i < cutoff_index) sorted_returns in
      List.fold_left (+.) 0. tail_returns /. float_of_int (List.length tail_returns)
  | ExpectedShortfall confidence ->
      let sorted_returns = List.sort compare returns in
      let cutoff_index = int_of_float (float_of_int (List.length returns) *. (1. -. confidence)) in
      let tail_returns = List.filteri (fun i _ -> i < cutoff_index) sorted_returns in
      List.fold_left (fun acc r -> acc +. abs_float r) 0. tail_returns /. float_of_int (List.length tail_returns)
  | MaxDrawdown ->
      let rec calc_max_drawdown peak drawdown = function
        | [] -> drawdown
        | hd :: tl ->
            if hd > peak then
              calc_max_drawdown hd drawdown tl
            else
              let current_drawdown = (peak -. hd) /. peak in
              calc_max_drawdown peak (max drawdown current_drawdown) tl
      in
      calc_max_drawdown 0. 0. returns

let check_risk_limit portfolio risk_management =
  let current_risk = calculate_risk portfolio risk_management.measure in
  match risk_management.limit with
  | AbsoluteLimit limit -> current_risk <= limit
  | RelativeLimit limit ->
      let portfolio_value = Portfolio.get_total_value portfolio in
      current_risk <= limit *. portfolio_value

let adjust_position portfolio risk_management =
  if not (check_risk_limit portfolio risk_management) then
    let current_weights = Portfolio.get_weights portfolio in
    let reduced_weights = Array.map (fun w -> w *. 0.9) current_weights in
    Portfolio.rebalance portfolio reduced_weights

let calculate_advanced_risk portfolio risk_measure =
  let returns = Portfolio.get_returns portfolio in
  match risk_measure with
  | TailValueAtRisk confidence ->
      let sorted_returns = List.sort compare returns in
      let cutoff_index = int_of_float (float_of_int (List.length returns) *. (1. -. confidence)) in
      let tail_returns = List.filteri (fun i _ -> i < cutoff_index) sorted_returns in
      List.fold_left (+.) 0. tail_returns /. float_of_int (List.length tail_returns)
  | ExpectedTail confidence ->
      let sorted_returns = List.sort compare returns in
      let cutoff_index = int_of_float (float_of_int (List.length returns) *. (1. -. confidence)) in
      let tail_returns = List.filteri (fun i _ -> i < cutoff_index) sorted_returns in
      List.fold_left (fun acc r -> acc +. (r -. List.hd tail_returns) ** 2.) 0. tail_returns
      /. float_of_int (List.length tail_returns)
  | ConditionalDrawdown threshold ->
      let max_drawdown = ref 0. in
      let current_drawdown = ref 0. in
      let peak = ref (List.hd returns) in
      List.iter (fun r ->
        if r > !peak then peak := r
        else
          let drawdown = (!peak -. r) /. !peak in
          if drawdown > !current_drawdown then current_drawdown := drawdown;
          if !current_drawdown > threshold then max_drawdown := max !max_drawdown !current_drawdown;
          if r = !peak then current_drawdown := 0.
      ) returns;
      !max_drawdown

let adjust_position_advanced portfolio risk_management =
  let risk = calculate_advanced_risk portfolio risk_management.measure in
  let threshold = match risk_management.limit with
    | AbsoluteLimit limit -> limit
    | RelativeLimit limit -> limit *. Portfolio.get_total_value portfolio
  in
  if risk > threshold then
    let current_weights = Portfolio.get_weights portfolio in
    let risk_contribution = calculate_risk_contribution portfolio in
    let total_risk = Array.fold_left (+.) 0. risk_contribution in
    let new_weights = Array.map2 (fun w rc ->
      w *. (1. -. (rc /. total_risk) *. 0.1)
    ) current_weights risk_contribution in
    Portfolio.rebalance portfolio new_weights

let calculate_risk_contribution portfolio =
  let returns = Portfolio.get_returns portfolio in
  let weights = Portfolio.get_weights portfolio in
  let covariance_matrix = calculate_covariance_matrix returns in
  let portfolio_variance = calculate_portfolio_variance weights covariance_matrix in
  Array.mapi (fun i w ->
    let marginal_contribution = Array.fold_left (fun acc j ->
      acc +. weights.(j) *. covariance_matrix.(i).(j)
    ) 0. weights in
    w *. marginal_contribution /. sqrt portfolio_variance
  ) weights

let calculate_covariance_matrix returns =
  let n = List.length returns in
  let means = List.map (fun r -> List.fold_left (+.) 0. r /. float_of_int (List.length r)) returns in
  Array.init n (fun i ->
    Array.init n (fun j ->
      let cov = List.fold_left2 (fun acc ri rj ->
        acc +. (ri -. List.nth means i) *. (rj -. List.nth means j)
      ) 0. (List.nth returns i) (List.nth returns j) in
      cov /. float_of_int (List.length (List.nth returns i) - 1)
    )
  )

let calculate_portfolio_variance weights covariance_matrix =
  Array.fold_left (fun acc_i i ->
    acc_i +. weights.(i) *. (Array.fold_left (fun acc_j j ->
      acc_j +. weights.(j) *. covariance_matrix.(i).(j)
    ) 0. weights)
  ) 0. weights