open Torch

type t = {
  cumulative_return: float;
  annualized_return: float;
  annualized_volatility: float;
  sharpe_ratio: float;
  sortino_ratio: float;
  maximum_drawdown: float;
  value_at_risk: float;
  conditional_value_at_risk: float;
  calmar_ratio: float;
  omega_ratio: float;
  information_ratio: float;
  treynor_ratio: float;
}

let calculate model portfolio initial_wealth risk_free_rate num_periods benchmark_opt =
  let returns = Array.init num_periods (fun _ -> 
    Nmvm.sample model |> fun r -> Tensor.(dot portfolio r |> to_float0)
  ) in
  let wealth = Array.make (num_periods + 1) initial_wealth in
  for i = 1 to num_periods do
    wealth.(i) <- wealth.(i-1) *. (1. +. returns.(i-1))
  done;

  let cumulative_return = (Array.last wealth /. initial_wealth) -. 1. in
  let annualized_return = (cumulative_return +. 1.) ** (252. /. Float.of_int num_periods) -. 1. in
  let annualized_volatility = Owl.Stats.std returns *. sqrt 252. in
  let sharpe_ratio = (annualized_return -. risk_free_rate) /. annualized_volatility in
  
  let downside_returns = Array.map (fun r -> min (r -. risk_free_rate) 0.) returns in
  let downside_volatility = Owl.Stats.std downside_returns *. sqrt 252. in
  let sortino_ratio = (annualized_return -. risk_free_rate) /. downside_volatility in
  
  let max_drawdown = Array.fold_left (fun (max_dd, peak) w ->
    if w > peak then (max_dd, w)
    else (max (peak -. w) /. peak, peak)
  ) (0., initial_wealth) wealth |> fst in
  
  let var = Owl.Stats.quantile returns 0.05 in
  let cvar = Array.fold_left (fun (sum, count) r ->
    if r < var then (sum +. r, count +. 1.)
    else (sum, count)
  ) (0., 0.) returns |> fun (sum, count) -> sum /. count in
  
  let calmar_ratio = annualized_return /. max_drawdown in
  
  let omega_ratio = 
    let threshold = risk_free_rate /. 252. in
    let (gains, losses) = Array.fold_left (fun (g, l) r ->
      if r > threshold then (g +. (r -. threshold), l)
      else (g, l +. (threshold -. r))
    ) (0., 0.) returns in
    gains /. losses
  in

  let (information_ratio, treynor_ratio) = match benchmark_opt with
    | Some benchmark ->
        let benchmark_returns = Array.init num_periods (fun _ -> 
          Nmvm.sample model |> fun r -> Tensor.(dot benchmark r |> to_float0)
        ) in
        let excess_returns = Array.map2 (-.) returns benchmark_returns in
        let ir = (Owl.Stats.mean excess_returns) /. (Owl.Stats.std excess_returns) in
        let beta = Owl.Stats.covariance returns benchmark_returns /. Owl.Stats.var benchmark_returns in
        let tr = (annualized_return -. risk_free_rate) /. beta in
        (ir, tr)
    | None -> (0., 0.)  (* Default values when no benchmark is provided *)
  in

  {
    cumulative_return;
    annualized_return;
    annualized_volatility;
    sharpe_ratio;
    sortino_ratio;
    maximum_drawdown = max_drawdown;
    value_at_risk = var;
    conditional_value_at_risk = cvar;
    calmar_ratio;
    omega_ratio;
    information_ratio;
    treynor_ratio;
  }

let to_string perf =
  Printf.sprintf
    "Cumulative Return: %.2f%%\n\
     Annualized Return: %.2f%%\n\
     Annualized Volatility: %.2f%%\n\
     Sharpe Ratio: %.2f\n\
     Sortino Ratio: %.2f\n\
     Maximum Drawdown: %.2f%%\n\
     Value at Risk (5%%): %.2f%%\n\
     Conditional Value at Risk (5%%): %.2f%%\n\
     Calmar Ratio: %.2f\n\
     Omega Ratio: %.2f\n\
     Information Ratio: %.2f\n\
     Treynor Ratio: %.2f"
    (perf.cumulative_return *. 100.)
    (perf.annualized_return *. 100.)
    (perf.annualized_volatility *. 100.)
    perf.sharpe_ratio
    perf.sortino_ratio
    (perf.maximum_drawdown *. 100.)
    (perf.value_at_risk *. 100.)
    (perf.conditional_value_at_risk *. 100.)
    perf.calmar_ratio
    perf.omega_ratio
    perf.information_ratio
    perf.treynor_ratio