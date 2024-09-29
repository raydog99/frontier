open Torch

type t = {
  name: string;
  factors: Factor.t list;
}

let create name factors =
  { name; factors }

let get_factor_by_name model name =
  List.find_opt (fun f -> f.name = name) model.factors

let factor_correlations model =
  let n = List.length model.factors in
  let corr_matrix = Tensor.zeros [n; n] in
  List.iteri (fun i factor1 ->
    List.iteri (fun j factor2 ->
      if i <= j then
        let corr = Util.correlation factor1.data factor2.data in
        Tensor.set corr_matrix [i; j] (Scalar.float corr);
        if i <> j then
          Tensor.set corr_matrix [j; i] (Scalar.float corr)
    ) model.factors
  ) model.factors;
  corr_matrix

let spanning_regression target_factor explanatory_factors =
  try
    let y = target_factor.data in
    let x = Tensor.stack (List.map (fun f -> f.data) explanatory_factors) ~dim:1 in
    let xt = Tensor.transpose x 0 1 in
    let xtx = Tensor.matmul xt x in
    let xty = Tensor.matmul xt y in
    let coeffs = Tensor.solve xtx xty in
    let y_pred = Tensor.matmul x coeffs in
    let residuals = Tensor.sub y y_pred in
    let alpha = Tensor.mean residuals ~dim:[0] ~keepdim:false in
    let n = Tensor.size y 0 in
    let k = Tensor.size x 1 in
    let df = n - k - 1 in
    let mse = Tensor.div (Tensor.sum (Tensor.pow residuals (Scalar.float 2.))) (Tensor.float (float_of_int df)) in
    let std_err = Tensor.sqrt (Tensor.mul mse (Tensor.diag (Tensor.inverse xtx))) in
    let t_stats = Tensor.div coeffs std_err in
    let p_values = Util.p_value t_stats (float_of_int df) in
    (coeffs, alpha, t_stats, p_values)
  with _ ->
    raise (Error.StatisticalError "Failed to perform spanning regression")

let grs_test model target_factors =
  try
    let n = Tensor.size (List.hd model.factors).data 0 in
    let k = List.length target_factors in
    let m = List.length model.factors in
    let alpha_matrix = Tensor.stack (List.map (fun f -> 
      let (_, alpha, _, _) = spanning_regression f model.factors in
      alpha
    ) target_factors) ~dim:0 in
    let sigma = Tensor.cov (Tensor.stack (List.map (fun f -> f.data) target_factors) ~dim:1) in
    let grs_statistic = Tensor.matmul (Tensor.matmul alpha_matrix (Tensor.inverse sigma)) (Tensor.transpose alpha_matrix ~dim0:0 ~dim1:1) in
    let grs_statistic = Tensor.mul grs_statistic (Tensor.float ((float_of_int (n - m - k)) /. (float_of_int (k * (n - m - 1))))) in
    let p_value = Util.p_value grs_statistic (float_of_int (k * (n - m - k))) in
    (grs_statistic, p_value)
  with _ ->
    raise (Error.StatisticalError "Failed to perform GRS test")

let compare_models model1 model2 config =
  let all_factors = List.append model1.factors model2.factors in
  let unique_factors = List.sort_uniq (fun f1 f2 -> String.compare f1.name f2.name) all_factors in
  List.map (fun factor ->
    let in_model1 = List.exists (fun f -> f.name = factor.name) model1.factors in
    let in_model2 = List.exists (fun f -> f.name = factor.name) model2.factors in
    let (mean, std, t_stat, p_val) = Factor.calculate_premium factor in
    let nw_t_stat = Factor.calculate_newey_west_t_stat factor config.newey_west_lags in
    let (ci_lower, ci_upper) = Factor.bootstrap_confidence_interval factor config.confidence_level config.n_bootstrap in
    (factor.name, in_model1, in_model2, mean, std, t_stat, p_val, nw_t_stat, ci_lower, ci_upper)
  ) unique_factors

let analyze_premiums_over_time model config =
  List.map (fun factor ->
    let yearly_data = Tensor.split factor.data ~split_size:12 ~dim:0 in
    let yearly_premiums = List.mapi (fun i year_data ->
      let year = config.start_year + i in
      let year_factor = Factor.create factor.name year_data in
      (year, Factor.calculate_premium year_factor)
    ) yearly_data in
    (factor.name, yearly_premiums)
  ) model.factors

let calculate_factor_loadings model data =
  try
    let x = Tensor.stack (List.map (fun f -> f.data) model.factors) ~dim:1 in
    let xt = Tensor.transpose x 0 1 in
    let xtx = Tensor.matmul xt x in
    let xty = Tensor.matmul xt data in
    Tensor.solve xtx xty
  with _ ->
    raise (Error.ModelError "Failed to calculate factor loadings")

let calculate_information_ratio model benchmark =
  let active_returns = Tensor.sub (Tensor.stack (List.map (fun f -> f.data) model.factors) ~dim:1) benchmark in
  let mean_active_return = Tensor.mean active_returns ~dim:[0] ~keepdim:false in
  let std_active_return = Tensor.std active_returns ~dim:[0] ~keepdim:false ~unbiased:true in
  Tensor.div mean_active_return std_active_return

let calculate_tracking_error model benchmark =
  let active_returns = Tensor.sub (Tensor.stack (List.map (fun f -> f.data) model.factors) ~dim:1) benchmark in
  Tensor.std active_returns ~dim:[0] ~keepdim:false ~unbiased:true

let cross_sectional_regression model dependent_var =
  let independent_vars = Tensor.stack (List.map (fun f -> f.data) model.factors) ~dim:1 in
  Util.fama_macbeth_regression dependent_var independent_vars

let rolling_factor_loadings model data config =
  Util.rolling_window data config.rolling_window_size config.rolling_window_step (fun window ->
    calculate_factor_loadings model window
  )

let calculate_factor_exposures model returns =
  let factor_returns = Tensor.stack (List.map (fun f -> f.data) model.factors) ~dim:1 in
  let xt = Tensor.transpose factor_returns 0 1 in
  let xtx = Tensor.matmul xt factor_returns in
  let xty = Tensor.matmul xt returns in
  Tensor.solve xtx xty

let perform_spanning_tests model1 model2 =
  let factors1 = Tensor.stack (List.map (fun f -> f.data) model1.factors) ~dim:1 in
  let factors2 = Tensor.stack (List.map (fun f -> f.data) model2.factors) ~dim:1 in
  
  let test_factor model factors =
    let (coeffs, alpha, t_stats, p_values) = spanning_regression (Factor.create "Test" factors) model.factors in
    let f_stat = Tensor.div (Tensor.mul alpha alpha) (Tensor.sum (Tensor.pow (Tensor.sub factors (Tensor.matmul factors coeffs)) (Scalar.float 2.))) in
    (alpha, t_stats, p_values, f_stat)
  in
  
  let results1 = test_factor model2 factors1 in
  let results2 = test_factor model1 factors2 in
  (results1, results2)

let calculate_factor_exposures_timeseries model returns window_size =
  let n = Tensor.size returns 0 in
  let k = List.length model.factors in
  let exposures = Tensor.zeros [n - window_size + 1; k] in
  for i = 0 to n - window_size do
    let window_returns = Tensor.narrow returns ~dim:0 ~start:i ~length:window_size in
    let window_factors = List.map (fun f -> Tensor.narrow f.data ~dim:0 ~start:i ~length:window_size) model.factors in
    let window_exposures = calculate_factor_exposures model window_returns in
    Tensor.copy_ (Tensor.select exposures 0 i) window_exposures;
  done;
  exposures

let calculate_cross_sectional_regression model returns =
  let factor_returns = Tensor.stack (List.map (fun f -> f.data) model.factors) ~dim:1 in
  Util.fama_macbeth_regression returns factor_returns