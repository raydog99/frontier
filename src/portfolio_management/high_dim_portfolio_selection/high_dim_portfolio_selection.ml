open Torch

type utility_type = 
  | Linear of float * float     
  | Quadratic of float * float  
  | Exponential of float * float 
  | Power of float * float      
  | Logarithmic of float        

type portfolio = {
  weights: Tensor.t;
  utility: utility_type;
  lambda: float;
  cardinality: int option;
}

type market_data = {
  prices: Tensor.t;
  volumes: Tensor.t;
  market_caps: Tensor.t;
  dates: int array;
  index_name: string;
}

type optimization_config = {
  max_iter: int;
  tol: float;
  screening_freq: int;
  step_size: float;
  min_active_set: int;
}

type performance_metrics = {
  returns: float;
  volatility: float;
  sharpe_ratio: float;
  max_drawdown: float;
  turnover: float;
  active_positions: int;
}

type validation_result = {
  error_bounds: float * float;
  confidence_level: float;
  hypothesis_tests: (string * float * float) list;
}

type market_stats = {
  correlations: Tensor.t;
  volatilities: Tensor.t;
  mean_correlation: float;
  mean_volatility: float;
  max_drawdown: float;
}

type robustness_result = {
  parameter_sensitivity: (float * performance_metrics) list;
  stability_measure: float;
  turnover_analysis: float list;
}

let default_optimization_config = {
  max_iter = 1000;
  tol = 1e-6;
  screening_freq = 10;
  step_size = 0.01;
  min_active_set = 10;
}

let default_kelly_params = {
  eta: 1e-6;
  min_weight: 1e-4;
}

let default_hara_params = {
  gamma: 0.5;
  eta: 1e-6;
  risk_aversion: 2.0;
}

let evaluate_utility util x =
  match util with
  | Linear (a, eta) -> 
      Tensor.(eta + (mul_scalar x a))
  | Quadratic (eta, lambda) -> 
      Tensor.(eta * x - mul_scalar (pow_scalar x 2.) (lambda /. 2.))
  | Exponential (a, eta) ->
      Tensor.(ones_like x - exp (neg (mul_scalar (add_scalar x eta) a)))
  | Power (gamma, eta) ->
      Tensor.(pow_scalar (add_scalar x eta) gamma)
  | Logarithmic eta ->
      Tensor.(log (add_scalar x eta))

let gradient_utility util x =
  match util with
  | Linear (a, _) -> 
      Tensor.(full_like x a)
  | Quadratic (eta, lambda) ->
      Tensor.(sub (full_like x eta) (mul_scalar x lambda))
  | Exponential (a, eta) ->
      Tensor.(mul_scalar (exp (neg (mul_scalar (add_scalar x eta) a))) a)
  | Power (gamma, eta) ->
      Tensor.(mul_scalar (pow_scalar (add_scalar x eta) (gamma -. 1.)) gamma)
  | Logarithmic eta ->
      Tensor.(div_scalar (ones_like x) (add_scalar x eta))

let l1_proximal tensor lambda =
  let thresholded = Tensor.(
    mul (gt tensor (full_like tensor lambda))
        (sub tensor (full_like tensor lambda))
  ) in
  thresholded

let project_simplex tensor =
  let sorted, _ = Tensor.sort tensor ~descending:true ~stable:true ~dim:0 in
  let n = Tensor.size tensor 0 in
  let cumsum = Tensor.cumsum sorted ~dim:0 ~dtype:(T Float) in
  let indices = Tensor.arange ~end_:(float_of_int n) ~options:(Device.Cpu, T Float) in
  let indices = Tensor.(add_scalar indices 1.) in
  let slopes = Tensor.(div sorted indices) in
  let crossing = Tensor.(gt slopes (div cumsum indices)) in
  let k = Tensor.(sum crossing ~dtype:(T Int64)) |> Tensor.int64_value |> Int64.to_int in
  let tau = Tensor.(div (get cumsum (k - 1)) (float_of_int k)) in
  Tensor.(relu (sub tensor tau))

let screen_features portfolio data =
  let n = Tensor.size data.prices 0 in
  let d = Tensor.size data.prices 1 in
  
  (* Compute dual variables *)
  let returns = Tensor.(mm data.prices portfolio.weights) in
  let grad = Tensor.(neg (gradient_utility portfolio.utility returns)) in
  let dual = Tensor.(mm (transpose data.prices 0 1) grad) in
  
  (* Screen features *)
  let active = ref [] in
  let screened = ref [] in
  
  for j = 0 to d - 1 do
    let feat_dual = Tensor.(select dual ~dim:0 ~index:j |> float_value) in
    if abs_float feat_dual <= portfolio.lambda then
      screened := j :: !screened
    else
      active := j :: !active
  done;
  
  (!active, !screened)

let compute_metrics returns =
  let n = Tensor.size returns 0 in
  
  let mean_return = Tensor.(mean returns |> float_value) in
  let volatility = 
    Tensor.(std returns ~dim:[0] ~unbiased:true |> float_value) in
  
  let sharpe = 
    if volatility > 0. then mean_return /. volatility else 0. in
  
  let cumsum = Tensor.cumsum returns ~dim:0 ~dtype:(T Float) in
  let running_max = Tensor.cummax cumsum ~dim:0 |> fst in
  let drawdowns = Tensor.(sub running_max cumsum) in
  let max_dd = Tensor.(max drawdowns ~keepdim:false |> float_value) in
  
  mean_return, volatility, sharpe, max_dd

let evaluate_performance portfolio data =
  let returns = Tensor.(mm data.prices portfolio.weights) in
  let mean_ret, vol, sharpe, max_dd = compute_metrics returns in
  
  let active = 
    Tensor.(sum (gt portfolio.weights (zeros_like portfolio.weights)))
    |> Tensor.int64_value
    |> Int64.to_int in
  
  {
    returns = mean_ret;
    volatility = vol;
    sharpe_ratio = sharpe;
    max_drawdown = max_dd;
    turnover = 0.;  (* Computed during rebalancing *)
    active_positions = active;
  }

let optimize ?(config=default_optimization_config) portfolio data =
  let n = Tensor.size data.prices 0 in
  let d = Tensor.size data.prices 1 in
  
  (* Initialize weights *)
  let init_weights = 
    Tensor.(div_scalar (ones [d]) (float_of_int d)) in
  
  let rec optimize_iter weights iter active_set prev_obj =
    if iter >= config.max_iter then 
      {portfolio with weights}
    else
      (* Compute gradient *)
      let returns = Tensor.(mm data.prices weights) in
      let full_grad = gradient_utility portfolio.utility returns in
      let grad = Tensor.(mm (transpose data.prices 0 1) full_grad) in
      
      (* Proximal gradient step *)
      let step = config.step_size /. sqrt (float_of_int iter +. 1.) in
      let new_weights = 
        Tensor.(sub weights (mul_scalar grad step))
        |> fun w -> l1_proximal w portfolio.lambda
        |> project_simplex in
      
      (* Update screening *)
      let active_set = 
        if iter mod config.screening_freq = 0 then
          let active, _ = screen_features 
            {portfolio with weights = new_weights} data in
          if List.length active < config.min_active_set then
            active_set
          else active
        else active_set in
      
      (* Compute objective *)
      let obj = 
        let new_returns = Tensor.(mm data.prices new_weights) in
        Tensor.(mean (evaluate_utility portfolio.utility new_returns) 
                |> float_value) in
      
      (* Check convergence *)
      if abs_float (obj -. prev_obj) < config.tol then
        {portfolio with weights = new_weights}
      else
        optimize_iter new_weights (iter + 1) active_set obj in
  
  optimize_iter init_weights 0 (List.init d (fun i -> i))
    (Tensor.float_value (
       evaluate_utility portfolio.utility 
         (Tensor.(mm data.prices init_weights))))

let create_kelly_portfolio default_kelly_params data =
  let d = Tensor.size data.prices 1 in
  
  let portfolio = {
    weights = Tensor.(div_scalar (ones [d]) (float_of_int d));
    utility = Logarithmic eta;
    lambda = 0.01;  (* Default regularization *)
    cardinality = None;
  } in
  
  let optimized = optimize portfolio data in
  
  (* Apply minimum weight constraint *)
  let thresholded = Tensor.(
    mul (gt optimized.weights (full_like optimized.weights min_weight))
        optimized.weights
  ) in
  let final_weights = 
    Tensor.(div thresholded (sum thresholded ~dim:[0])) in
  
  {optimized with weights = final_weights}

let create_hara_portfolio default_hara_params data =
  let d = Tensor.size data.prices 1 in
  
  let portfolio = {
    weights = Tensor.(div_scalar (ones [d]) (float_of_int d));
    utility = Power (gamma, eta);
    lambda = risk_aversion *. 0.01;  (* Scale regularization by risk aversion *)
    cardinality = None;
  } in
  
  optimize portfolio data

let validate_portfolio portfolio data ?(confidence_level=0.95) =
  let n = Tensor.size data.prices 0 in
  let returns = Tensor.(mm data.prices portfolio.weights) in
  
  (* Compute error bounds using bootstrap *)
  let num_bootstrap = 1000 in
  let bootstrap_returns = ref [] in
  
  for _ = 1 to num_bootstrap do
    let indices = 
      List.init n (fun _ -> Random.int n)
      |> Array.of_list
      |> Tensor.of_int1 in
    let sample = Tensor.index_select returns ~dim:0 ~index:indices in
    let mean_ret = Tensor.(mean sample |> float_value) in
    bootstrap_returns := mean_ret :: !bootstrap_returns
  done;
  
  let sorted_returns = 
    List.sort compare !bootstrap_returns in
  let lower_idx = 
    int_of_float (float_of_int num_bootstrap *. 
                 (1. -. confidence_level) /. 2.) in
  let upper_idx = 
    int_of_float (float_of_int num_bootstrap *. 
                 (1. +. confidence_level) /. 2.) in
  
  (* Compute t-statistic *)
  let mean_ret = Tensor.(mean returns |> float_value) in
  let std_ret = 
    Tensor.(std returns ~dim:[0] ~unbiased:true |> float_value) in
  let t_stat = mean_ret /. (std_ret /. sqrt (float_of_int n)) in
  let p_value = 
    2. *. (1. -. Statistics.standard_normal_cdf (abs_float t_stat)) in
  
  {
    error_bounds = 
      (List.nth sorted_returns lower_idx, 
       List.nth sorted_returns upper_idx);
    confidence_level;
    hypothesis_tests = [("t-test", t_stat, p_value)];
  }

let analyze_market data =
  let n = Tensor.size data.prices 0 in
  let returns = Tensor.(
    log (div (narrow data.prices ~dim:0 ~start:1 ~length:(n-1))
             (narrow data.prices ~dim:0 ~start:0 ~length:(n-1)))
  ) in
  
  let mean_rets = Tensor.mean returns ~dim:[0] ~keepdim:true in
  let centered = Tensor.(sub returns mean_rets) in
  let std_rets = 
    Tensor.(std centered ~dim:[0] ~keepdim:true ~unbiased:true) in
  let normalized = Tensor.(div centered std_rets) in
  let correlations = Tensor.(
    mm (transpose normalized 0 1) normalized
    |> div_scalar (float_of_int (n - 1))
  ) in
  
  let volatilities = 
    Tensor.(std returns ~dim:[0] ~unbiased:true)
    |> Tensor.mul_scalar (sqrt 252.) in
  
  let cum_returns = Tensor.cumsum returns ~dim:0 ~dtype:(T Float) in
  let running_max = Tensor.cummax cum_returns ~dim:0 |> fst in
  let drawdowns = Tensor.(sub running_max cum_returns) in
  let max_dd = Tensor.(max drawdowns ~keepdim:false |> float_value) in
  
  {
    correlations;
    volatilities;
    mean_correlation = 
      Tensor.(mean (triu correlations ~diagonal:1) |> float_value);
    mean_volatility = 
      Tensor.(mean volatilities |> float_value);
    max_drawdown = max_dd;
  }

let rebalance_portfolio portfolio data ?(transaction_costs=true) =
  let n = Tensor.size data.prices 0 in
  let d = Tensor.size data.prices 1 in
  
  let new_portfolio = optimize portfolio data in
  
  let turnover = Tensor.(
    sum (abs (sub new_portfolio.weights portfolio.weights))
    |> float_value
  ) in
  
  let final_weights =
    if transaction_costs then
      let cost_factor = 0.001 *. turnover in  (* 10 bps per turnover *)
      Tensor.(mul_scalar new_portfolio.weights (1. -. cost_factor))
      |> project_simplex
    else
      new_portfolio.weights in
  
  let returns = Tensor.(mm data.prices final_weights) in
  let mean_ret, vol, sharpe, max_dd = compute_metrics returns in
  
  let metrics = {
    returns = mean_ret;
    volatility = vol;
    sharpe_ratio = sharpe;
    max_drawdown = max_dd;
    turnover;
    active_positions = 
      Tensor.(sum (gt final_weights (zeros_like final_weights)))
      |> Tensor.int64_value
      |> Int64.to_int;
  } in
  
  ({new_portfolio with weights = final_weights}, metrics)

let cross_validate portfolio data num_folds =
  let n = Tensor.size data.prices 0 in
  let fold_size = n / num_folds in
  
  let metrics_list = ref [] in
  
  for fold = 0 to num_folds - 1 do
    let test_start = fold * fold_size in
    let test_end = min (test_start + fold_size) n in
    
    let train_prices = Tensor.cat [
      Tensor.narrow data.prices ~dim:0 ~start:0 ~length:test_start;
      Tensor.narrow data.prices ~dim:0 
        ~start:test_end ~length:(n - test_end);
    ] ~dim:0 in
    
    let test_prices = 
      Tensor.narrow data.prices 
        ~dim:0 ~start:test_start ~length:(test_end - test_start) in
    
    let train_data = {data with prices = train_prices} in
    let trained_portfolio = optimize portfolio train_data in
    
    let test_data = {data with prices = test_prices} in
    let metrics = evaluate_performance trained_portfolio test_data in
    metrics_list := metrics :: !metrics_list
  done;
  
  let metrics = List.rev !metrics_list in
  let mean_sharpe = 
    List.fold_left (fun acc m -> acc +. m.sharpe_ratio) 0. metrics /.
    float_of_int num_folds in
  
  let std_sharpe =
    let squared_diffs =
      List.fold_left 
        (fun acc m -> 
           acc +. (m.sharpe_ratio -. mean_sharpe) ** 2.) 
        0. metrics in
    sqrt (squared_diffs /. float_of_int (num_folds - 1)) in
  
  (metrics, mean_sharpe, std_sharpe)

let check_robustness portfolio data =
  (* Parameter sensitivity analysis *)
  let lambda_range = [0.001; 0.01; 0.1; 1.0] in
  let sensitivity = 
    List.map (fun lambda ->
      let test_portfolio = {portfolio with lambda} in
      let result = optimize test_portfolio data in
      let metrics = evaluate_performance result data in
      (lambda, metrics)
    ) lambda_range in
  
  (* Stability analysis *)
  let n = Tensor.size data.prices 0 in
  let window_size = n / 4 in  (* Quarter of data *)
  let stability_measures = ref [] in
  
  for i = 0 to 3 do
    let start_idx = i * window_size in
    let window_prices = 
      Tensor.narrow data.prices 
        ~dim:0 ~start:start_idx ~length:window_size in
    
    let window_data = {data with prices = window_prices} in
    let window_portfolio = optimize portfolio window_data in
    
    if i > 0 then
      let turnover = Tensor.(
        sum (abs (sub window_portfolio.weights portfolio.weights))
        |> float_value
      ) in
      stability_measures := turnover :: !stability_measures
  done;
  
  let stability = 
    List.fold_left (+.) 0. !stability_measures /. 3. in
  
  {
    parameter_sensitivity = sensitivity;
    stability_measure = stability;
    turnover_analysis = List.rev !stability_measures;
  }

let compare_strategies portfolios data =
  List.mapi (fun i portfolio ->
    let name = "Strategy_" ^ string_of_int (i + 1) in
    let optimized = optimize portfolio data in
    let metrics = evaluate_performance optimized data in
    (name, metrics)
  ) portfolios
  |> List.sort (fun (_, m1) (_, m2) -> 
       compare m2.sharpe_ratio m1.sharpe_ratio
     )

let prepare_market_data ~prices ~volumes ~market_caps ~dates ~index_name =
  (* Validate dimensions *)
  let n = Tensor.size prices 0 in
  let d = Tensor.size prices 1 in
  assert (Tensor.size volumes 0 = n && Tensor.size volumes 1 = d);
  assert (Tensor.size market_caps 0 = d);
  assert (Array.length dates = n);
  
  let clean_prices = 
    let mask = Tensor.(gt prices (zeros_like prices)) in
    let cleaned = Tensor.clone prices in
    
    (* Forward fill missing values *)
    for i = 1 to n - 1 do
      let prev_row = Tensor.select cleaned ~dim:0 ~index:(i-1) in
      let curr_row = Tensor.select cleaned ~dim:0 ~index:i in
      let curr_mask = Tensor.select mask ~dim:0 ~index:i in
      
      Tensor.(copy_ 
        ~src:(mul curr_mask curr_row + mul (lt curr_mask (ones_like curr_mask)) prev_row)
        ~dst:(select cleaned ~dim:0 ~index:i))
    done;
    cleaned in
  
  (* Handle extreme values *)
  let winsorized_prices =
    let q01 = Tensor.quantile prices ~q:0.01 ~dim:0 in
    let q99 = Tensor.quantile prices ~q:0.99 ~dim:0 in
    Tensor.(
      min (max clean_prices q01) q99
    ) in
  
  {
    prices = winsorized_prices;
    volumes;
    market_caps;
    dates;
    index_name;
  }