open Torch

type asset_data = {
  returns: Tensor.t;
  scales: float array;
  names: string array;
}

type portfolio_params = {
  target_hurst: float;
  min_weight: float;
  max_weight: float;
}

type scaling_params = {
  beta: float;
  alpha: float;
}

type multifractal_params = {
  q_min: float;
  q_max: float;
  q_steps: int;
  scale_min: float;
  scale_max: float;
  scale_steps: int;
}

type diffusion_params = {
  beta: float;
  alpha: float;
  k_alpha: float;
}

type performance_metrics = {
  sharpe_ratio: float;
  sortino_ratio: float;
  max_drawdown: float;
  annualized_return: float;
  annualized_vol: float;
  information_ratio: float;
}

type backtest_params = {
  lookback_days: int;
  rebalance_frequency: int;
  transaction_cost: float;
  start_date: float;
  end_date: float;
}

let compute_returns prices =
  let log_prices = log prices in
  sub (slice log_prices ~dim:0 ~start:1 ~end':None ~step:1)
      (slice log_prices ~dim:0 ~start:0 ~end':(-1) ~step:1)

let compute_variance tensor ~dim =
  let mean = mean tensor ~dim:[dim] ~keepdim:true in
  let centered = sub tensor mean in
  mean (mul centered centered) ~dim:[dim] ~keepdim:true

let compute_covariance x y =
  let mean_x = mean x ~dim:[0] ~keepdim:true in
  let mean_y = mean y ~dim:[0] ~keepdim:true in
  let centered_x = sub x mean_x in
  let centered_y = sub y mean_y in
  mean (mul centered_x centered_y) ~dim:[0]

let compute_correlation x y =
  let cov = compute_covariance x y in
  let std_x = sqrt (compute_variance x ~dim:0) in
  let std_y = sqrt (compute_variance y ~dim:0) in
  div cov (mul std_x std_y)

let binomial n k =
  let rec aux acc n k =
    if k = 0 then acc
    else aux (acc * n / k) (n - 1) (k - 1)
  in
  if k > n then 0
  else aux 1 n k

let compute_multiscale_covariance data ~scales =
  let n_assets = size data.returns |> Array.get 1 in
  let n_scales = Array.length scales in
  let cov_tensor = zeros [n_scales; n_assets; n_assets] in
  
  Array.iteri (fun i scale ->
    let shaped_returns = reshape data.returns ~shape:[(-1); int_of_float scale; n_assets] in
    let scale_cov = zeros [n_assets; n_assets] in
    
    for j = 0 to n_assets - 1 do
      for k = 0 to n_assets - 1 do
        let cov = compute_covariance 
          (select shaped_returns ~dim:2 ~index:j)
          (select shaped_returns ~dim:2 ~index:k) in
        index_put_ scale_cov ~indices:[|j; k|] ~values:cov
      done
    done;
    
    index_put_ cov_tensor ~indices:[|i|] ~values:scale_cov
  ) scales;
  cov_tensor

let compute_hurst_exponent returns ~scales =
  let variances = zeros [Array.length scales] in
  
  Array.iteri (fun i scale ->
    let scaled_returns = reshape returns ~shape:[(-1); int_of_float scale] in
    let var = compute_variance scaled_returns ~dim:1 in
    index_put_ variances ~indices:[|i|] ~values:var
  ) scales;
  
  let log_scales = Tensor.of_float1 (Array.map log scales) in
  let log_vars = log variances in
  let slope = compute_covariance log_scales log_vars in
  float_value slope /. 2.0

let optimize_min_variance ?(constraints=None) covariance =
  let n_assets = size covariance |> Array.get 1 in
  let reg_cov = add covariance (mul (eye n_assets) (scalar 1e-6)) in
  let ones = ones [n_assets; 1] in
  let inv_cov = inverse reg_cov in
  let numerator = matmul inv_cov ones in
  let denominator = sum (matmul (transpose ones) numerator) in
  div numerator denominator

let optimize_max_sharpe ?(constraints=None) covariance returns =
  let n_assets = size covariance |> Array.get 1 in
  let mean_returns = mean returns ~dim:[0] |> reshape ~shape:[n_assets; 1] in
  let reg_cov = add covariance (mul (eye n_assets) (scalar 1e-6)) in
  let inv_cov = inverse reg_cov in
  let numerator = matmul inv_cov mean_returns in
  let denominator = sum (matmul (transpose mean_returns) numerator) in
  div numerator denominator

let compute_mad tensor ~dim 
  let median_val = median tensor ~dim:[dim] ~keepdim:true in
  let abs_dev = abs (sub tensor median_val) in
  median abs_dev ~dim:[dim] ~keepdim:true

let compute_l1_covariance returns 
  let n_assets = size returns |> Array.get 1 in
  let result = zeros [n_assets; n_assets] in
  
  let medians = median returns ~dim:[0] ~keepdim:true in
  let mads = compute_mad returns ~dim:0 in
  
  for i = 0 to n_assets - 1 do
    for j = 0 to n_assets - 1 do
      let dev_i = abs (sub (select returns ~dim:1 ~index:i) 
                          (select medians ~dim:1 ~index:i)) in
      let dev_j = abs (sub (select returns ~dim:1 ~index:j)
                          (select medians ~dim:1 ~index:j)) in
      
      let corr = if i = j then scalar 1.0
                else compute_correlation dev_i dev_j in
      
      let l1_cov = mul (mul (get mads ~index:i) (get mads ~index:j)) corr in
      index_put_ result ~indices:[|i; j|] ~values:l1_cov
    done
  done;
  result

let compute_robust_multiscale_covariance data ~scales 
  let n_assets = size data.returns |> Array.get 1 in
  let n_scales = Array.length scales in
  let ms_cov = zeros [n_scales; n_assets; n_assets] in
  
  Array.iteri (fun i scale ->
    let scaled_shape = [(-1); int_of_float scale; n_assets] in
    let scaled_returns = reshape data.returns ~shape:scaled_shape in
    let scale_cov = compute_l1_covariance scaled_returns in
    index_put_ ms_cov ~indices:[|i|] ~values:scale_cov
  ) scales;
  ms_cov

let gamma z =
  let coeffs = [|1.0; 0.5772156649; 0.0; -0.0001641725; 0.0; -0.0001643807|] in
  let result = ref (scalar 1.0) in
  let z_tensor = scalar z in
  let z_power = ref (scalar 1.0) in
  
  Array.iteri (fun i c ->
    z_power := mul !z_power z_tensor;
    result := add !result (mul (scalar c) !z_power)
  ) coeffs;
  !result

let caputo_derivative f params =
  let n = size f |> Array.get 0 in
  let result = zeros [n] in
  
  for i = 1 to n - 1 do
    let sum = ref (scalar 0.0) in
    for k = 0 to i - 1 do
      let weight = div (gamma (scalar (1.0 -. params.beta)))
                      (mul (gamma (scalar (float_of_int (k + 1))))
                           (gamma (scalar (1.0 -. params.beta -. float_of_int k)))) in
      let diff = sub (get f ~index:(i-k)) (get f ~index:(i-k-1)) in
      sum := add !sum (mul weight diff)
    done;
    
    let scaled_sum = div !sum (pow (scalar 1e-3) (scalar params.beta)) in
    index_put_ result ~indices:[|i|] ~values:scaled_sum
  done;
  result

let riesz_derivative f params =
  let n = size f |> Array.get 0 in
  let result = zeros [n] in
  
  for i = 2 to n - 3 do
    let sum = ref (scalar 0.0) in
    for k = (-2) to 2 do
      if k <> 0 then begin
        let coeff = scalar (
          (if k mod 2 = 0 then -1.0 else 1.0) *.
          float (binomial 4 (k + 2)) /.
          float_of_int (abs k) ** params.alpha
        ) in
        let value = get f ~index:(i + k) in
        sum := add !sum (mul coeff value)
      end
    done;
    
    let scaled_sum = div !sum (pow (scalar 1e-3) (scalar params.alpha)) in
    index_put_ result ~indices:[|i|] ~values:scaled_sum
  done;
  result

let solve_fractional_pde initial_condition params =
  let n_steps = 1000 in
  let n_points = size initial_condition |> Array.get 0 in
  let solution = zeros [n_steps; n_points] in
  
  index_put_ solution ~indices:[|0|] ~values:initial_condition;
  
  for t = 1 to n_steps - 1 do
    let prev = select solution ~dim:0 ~index:(t-1) in
    let time_deriv = caputo_derivative (narrow solution ~dim:0 ~start:0 ~length:t) params in
    let space_deriv = riesz_derivative prev params in
    let next = add prev (mul (scalar params.k_alpha) (mul time_deriv space_deriv)) in
    index_put_ solution ~indices:[|t|] ~values:next
  done;
  solution

let compute_rolling_returns returns window_size =
  let n = size returns |> Array.get 0 in
  let rolled = zeros [n - window_size + 1] in
  
  for i = 0 to n - window_size do
    let window = narrow returns ~dim:0 ~start:i ~length:window_size in
    let cumret = exp (sum window) |> sub (scalar 1.0) in
    index_put_ rolled ~indices:[|i|] ~values:cumret
  done;
  rolled

let compute_metrics returns rf_rate =
  let mean_ret = mean returns in
  let std_ret = std returns ~dim:None ~unbiased:true in
  let sharpe = div (sub mean_ret rf_rate) std_ret |> float_value in
  
  let downside_returns = maximum (neg returns) (scalar 0.0) in
  let downside_std = std downside_returns ~dim:None ~unbiased:true in
  let sortino = div (sub mean_ret rf_rate) downside_std |> float_value in
  
  let cumulative_returns = exp (cumsum returns ~dim:0) in
  let rolling_max = fold_right max cumulative_returns ~init:(scalar Float.neg_infinity) in
  let drawdowns = div (sub rolling_max cumulative_returns) rolling_max in
  let max_drawdown = max drawdowns |> float_value in
  
  {
    sharpe_ratio = sharpe;
    sortino_ratio = sortino;
    max_drawdown = max_drawdown;
    annualized_return = float_value mean_ret *. 252.0;
    annualized_vol = float_value std_ret *. sqrt 252.0;
    information_ratio = sharpe;
  }

let rebalance_portfolio data params current_weights current_date =
  let lookback_start = current_date -. (float_of_int params.lookback_days *. 86400.0) in
  let historical_data = {
    data with
    returns = narrow data.returns ~dim:0 
                    ~start:(int_of_float lookback_start) 
                    ~length:params.lookback_days
  } in
  
  let multiscale_cov = c_multiscale_covariance historical_data 
                        ~scales:data.scales in
  
  let new_weights = optimize_min_variance multiscale_cov in
  
  let weight_diff = sub new_weights current_weights in
  let transaction_cost = mul (abs weight_diff) (scalar params.transaction_cost) in
  sub new_weights transaction_cost

let run_backtest data params =
  let n_days = size data.returns |> Array.get 0 in
  let n_assets = size data.returns |> Array.get 1 in
  
  let weights = ones [n_assets] |> div (scalar (float_of_int n_assets)) in
  let portfolio_values = zeros [n_days] in
  let portfolio_returns = zeros [n_days - 1] in
  
  let rebalance_dates = ref [] in
  let weight_history = ref [] in
  
  for t = 0 to n_days - 2 do
    if t mod params.rebalance_frequency = 0 then begin
      let current_date = params.start_date +. (float_of_int t *. 86400.0) in
      let new_weights = rebalance_portfolio data params weights current_date in
      weights <- new_weights;
      rebalance_dates := current_date :: !rebalance_dates;
      weight_history := (copy weights) :: !weight_history
    end;
    
    let day_return = sum (mul weights (select data.returns ~dim:0 ~index:t)) in
    index_put_ portfolio_returns ~indices:[|t|] ~values:day_return;
    
    let port_value = if t = 0 then scalar 1.0 
                    else mul (get portfolio_values ~index:(t-1)) (exp day_return) in
    index_put_ portfolio_values ~indices:[|t|] ~values:port_value
  done;
  
  let metrics = compute_metrics portfolio_returns (scalar 0.02) in
  (portfolio_values, portfolio_returns, !weight_history, !rebalance_dates, metrics)

module StandardizedHurst = struct
  type hurst_components = {
    beta_n: float array;
    alpha_n: float array;
    h_n: float array;
  }

  let compute_standardized_hurst returns moments =
    let n_moments = Array.length moments in
    let result = {
      beta_n = Array.make n_moments 0.0;
      alpha_n = Array.make n_moments 0.0;
      h_n = Array.make n_moments 0.0;
    } in
    
    Array.iteri (fun i q ->
      let time_scales = [|1.; 5.; 10.; 21.|] in
      let time_fluctuations = zeros [Array.length time_scales] in
      
      Array.iteri (fun j scale ->
        let scaled_returns = reshape returns ~shape:[(-1); int_of_float scale] in
        let abs_returns = abs scaled_returns in
        let q_moment = pow abs_returns (scalar q) in
        let mean_q = mean q_moment ~dim:[1] in
        index_put_ time_fluctuations ~indices:[|j|] ~values:(log mean_q)
      ) time_scales;
      
      let log_scales = Tensor.of_float1 (Array.map log time_scales) in
      let beta = compute_covariance log_scales time_fluctuations |> float_value in
      result.beta_n.(i) <- beta;
      
      let diffs = sub (slice returns ~dim:0 ~start:1 ~end':None ~step:1)
                     (slice returns ~dim:0 ~start:0 ~end':(-1) ~step:1) in
      let sorted_diffs = sort (abs diffs) ~dim:0 ~descending:true in
      let n = size sorted_diffs |> Array.get 0 in
      let ranks = Tensor.of_float1 (Array.init n (fun j -> log (1.0 +. float_of_int j))) in
      let log_diffs = log sorted_diffs in
      
      let alpha = compute_covariance ranks log_diffs |> float_value |> (fun x -> -1.0 /. x) in
      result.alpha_n.(i) <- alpha;
      result.h_n.(i) <- beta /. alpha
    ) moments;
    result

  let test_multifractality h_components =
    let h1 = h_components.h_n.(0) in
    let h2 = h_components.h_n.(1) in
    abs_float (h2 -. 2.0 *. h1) > 0.05
end

module ScaleCorrelator = struct
  type scale_coupling = {
    correlation_hurst: float;
    volatility_feedback: float;
    scale_dependence: float array;
    cross_correlations: Tensor.t;
  }

  let compute_correlation_hurst returns scales =
    let n_assets = size returns |> Array.get 1 in
    let correlations = zeros [n_assets; n_assets] in
    let hursts = zeros [n_assets] in
    
    for i = 0 to n_assets - 1 do
      for j = i + 1 to n_assets - 1 do
        let corr = compute_correlation 
          (select returns ~dim:1 ~index:i)
          (select returns ~dim:1 ~index:j) in
        index_put_ correlations ~indices:[|i; j|] ~values:corr;
        index_put_ correlations ~indices:[|j; i|] ~values:corr
      done;
      
      let asset_returns = select returns ~dim:1 ~index:i in
      let hurst = compute_hurst_exponent asset_returns ~scales in
      index_put_ hursts ~indices:[|i|] ~values:(scalar hurst)
    done;
    
    compute_correlation (reshape correlations ~shape:[-1])
                            (reshape hursts ~shape:[-1]) |>
    float_value

  let analyze_scale_dependence returns scales =
    let n_scales = Array.length scales in
    let scale_returns = Array.map (fun scale ->
      let scale_int = int_of_float scale in
      reshape returns ~shape:[(-1); scale_int; size returns |> Array.get 1] |>
      mean ~dim:[1]
    ) scales in
    
    let cross_corr = zeros [n_scales; n_scales] in
    Array.iteri (fun i r1 ->
      Array.iteri (fun j r2 ->
        let corr = compute_correlation r1 r2 in
        index_put_ cross_corr ~indices:[|i; j|] ~values:corr
      ) scale_returns
    ) scale_returns;
    
    cross_corr

  let compute_scale_effects returns scales =
    let corr_hurst = compute_correlation_hurst returns scales in
    let cross_corr = analyze_scale_dependence returns scales in
    let n_scales = Array.length scales in
    let scale_deps = Array.init n_scales (fun i ->
      let scale_slice = select cross_corr ~dim:0 ~index:i in
      mean scale_slice |> float_value
    ) in
    
    {
      correlation_hurst = corr_hurst;
      volatility_feedback = mean cross_corr |> float_value;
      scale_dependence = scale_deps;
      cross_correlations = cross_corr;
    }

  let adjust_portfolio_weights weights effects =
    let coupling_adj = 
      if abs_float effects.correlation_hurst > 0.5 then 0.8 else 1.0 in
    let feedback_adj =
      if abs_float effects.volatility_feedback > 0.3 then 0.9 else 1.0 in
    
    let adjusted = mul weights (scalar (coupling_adj *. feedback_adj)) in
    div adjusted (sum adjusted)
end