open Torch

let tensor_to_float t =
  Tensor.to_float0_exn t

let correlation x y =
  let x_mean = Tensor.mean x ~dim:[0] ~keepdim:true in
  let y_mean = Tensor.mean y ~dim:[0] ~keepdim:true in
  let x_centered = Tensor.sub x x_mean in
  let y_centered = Tensor.sub y y_mean in
  let numerator = Tensor.sum (Tensor.mul x_centered y_centered) in
  let denominator = Tensor.sqrt (Tensor.mul (Tensor.sum (Tensor.pow x_centered (Scalar.float 2.)))
                                            (Tensor.sum (Tensor.pow y_centered (Scalar.float 2.))))
  in
  Tensor.div numerator denominator |> tensor_to_float

let t_statistic mean std n =
  Tensor.div mean (Tensor.div std (Tensor.sqrt (Tensor.float (float_of_int n))))

let p_value t_stat df =
  let abs_t = Tensor.abs t_stat in
  Tensor.sub (Tensor.float 1.) (Tensor.erf (Tensor.div abs_t (Tensor.sqrt (Tensor.float 2.))))

let calculate_irr cash_flows =
  let rec irr_helper rate =
    let npv = Tensor.sum (Tensor.div cash_flows (Tensor.pow (Tensor.add (Tensor.float 1.) rate) (Tensor.arange (Tensor.size cash_flows 0)))) in
    if Tensor.abs npv < Tensor.float 0.000001 then
      rate
    else
      let derivative = Tensor.sum (Tensor.div (Tensor.mul cash_flows (Tensor.arange (Tensor.size cash_flows 0)))
                                  (Tensor.pow (Tensor.add (Tensor.float 1.) rate) (Tensor.add (Tensor.arange (Tensor.size cash_flows 0)) (Tensor.float 1.)))) in
      irr_helper (Tensor.sub rate (Tensor.div npv derivative))
  in
  irr_helper (Tensor.float 0.1)

let newey_west_adjustment data lags =
  let n = Tensor.size data 0 in
  let autocovariances = List.init (lags + 1) (fun lag ->
    let x = Tensor.narrow data ~dim:0 ~start:0 ~length:(n - lag) in
    let y = Tensor.narrow data ~dim:0 ~start:lag ~length:(n - lag) in
    Tensor.mean (Tensor.mul x y) ~dim:[0] ~keepdim:false
  ) in
  let weights = List.init (lags + 1) (fun lag ->
    1. -. (float_of_int lag) /. (float_of_int (lags + 1))
  ) in
  let weighted_sum = List.fold_left2 (fun acc cov w ->
    Tensor.add acc (Tensor.mul cov (Tensor.float w))
  ) (Tensor.float 0.) autocovariances weights in
  Tensor.mul weighted_sum (Tensor.float 2.)

let bootstrap_confidence_interval data confidence_level n_bootstrap =
  let n = Tensor.size data 0 in
  let bootstrap_means = List.init n_bootstrap (fun _ ->
    let indices = Tensor.randint ~high:n ~size:[n] 0 in
    let sample = Tensor.index_select data ~dim:0 ~index:indices in
    Tensor.mean sample ~dim:[0] ~keepdim:false
  ) in
  let sorted_means = List.sort (fun a b -> compare (tensor_to_float a) (tensor_to_float b)) bootstrap_means in
  let lower_index = int_of_float (float_of_int n_bootstrap *. (1. -. confidence_level) /. 2.) in
  let upper_index = int_of_float (float_of_int n_bootstrap *. (1. +. confidence_level) /. 2.) in
  (List.nth sorted_means lower_index, List.nth sorted_means upper_index)

let rolling_window data window_size step_size f =
  let n = Tensor.size data 0 in
  let num_windows = (n - window_size) / step_size + 1 in
  List.init num_windows (fun i ->
    let start = i * step_size in
    let window = Tensor.narrow data ~dim:0 ~start ~length:window_size in
    f window
  )

let hansen_jagannathan_distance returns factor_returns =
  let excess_returns = Tensor.sub returns (Tensor.mean returns ~dim:[0] ~keepdim:true) in
  let factor_excess_returns = Tensor.sub factor_returns (Tensor.mean factor_returns ~dim:[0] ~keepdim:true) in
  let cov_matrix = Tensor.matmul (Tensor.transpose factor_excess_returns 0 1) factor_excess_returns in
  let inv_cov_matrix = Tensor.inverse cov_matrix in
  let beta = Tensor.matmul (Tensor.matmul excess_returns (Tensor.transpose factor_excess_returns 0 1)) inv_cov_matrix in
  let residuals = Tensor.sub excess_returns (Tensor.matmul beta factor_excess_returns) in
  let hj_distance = Tensor.sqrt (Tensor.mean (Tensor.pow residuals (Scalar.float 2.)) ~dim:[0] ~keepdim:false) in
  hj_distance

let gmm_estimation returns factor_returns n_iterations =
  let n = Tensor.size returns 0 in
  let k = Tensor.size factor_returns 1 in
  let initial_beta = Tensor.zeros [k; 1] in
  let rec gmm_iter beta iter =
    if iter >= n_iterations then beta
    else
      let residuals = Tensor.sub returns (Tensor.matmul factor_returns beta) in
      let moment_conditions = Tensor.matmul (Tensor.transpose factor_returns 0 1) residuals in
      let weight_matrix = Tensor.inverse (Tensor.matmul (Tensor.transpose moment_conditions 0 1) moment_conditions) in
      let new_beta = Tensor.matmul
        (Tensor.matmul
          (Tensor.inverse (Tensor.matmul (Tensor.matmul (Tensor.transpose factor_returns 0 1) weight_matrix) factor_returns))
          (Tensor.matmul (Tensor.transpose factor_returns 0 1) weight_matrix))
        returns in
      gmm_iter new_beta (iter + 1)
  in
  gmm_iter initial_beta 0

let calculate_turnover factor =
  let abs_diff = Tensor.abs (Tensor.sub factor.data (Tensor.roll factor.data 1 ~dims:[0] ~steps:1)) in
  Tensor.mean abs_diff

let calculate_information_coefficient factor returns =
  let ic = correlation factor.data returns in
  let n = Tensor.size factor.data 0 in
  let t_stat = Tensor.mul ic (Tensor.sqrt (Tensor.float (float_of_int (n - 2))))
               |> Tensor.div (Tensor.sqrt (Tensor.sub (Tensor.float 1.) (Tensor.mul ic ic))) in
  let p_value = p_value t_stat (float_of_int (n - 2)) in
  (ic, t_stat, p_value)

let calculate_autocorrelation factor lag =
  let n = Tensor.size factor.data 0 in
  let x = Tensor.narrow factor.data ~dim:0 ~start:0 ~length:(n - lag) in
  let y = Tensor.narrow factor.data ~dim:0 ~start:lag ~length:(n - lag) in
  correlation x y

let fama_macbeth_regression dependent_var independent_vars =
  let n_periods = Tensor.size dependent_var 0 in
  let n_vars = Tensor.size independent_vars 1 in
  
  let cross_sectional_regressions = Tensor.zeros [n_periods; n_vars] in
  for t = 0 to n_periods - 1 do
    let y_t = Tensor.select dependent_var 0 t in
    let x_t = Tensor.select independent_vars 0 t in
    let coeffs = Tensor.solve (Tensor.matmul (Tensor.transpose x_t 0 1) x_t) (Tensor.matmul (Tensor.transpose x_t 0 1) y_t) in
    Tensor.copy_ (Tensor.select cross_sectional_regressions 0 t) coeffs
  done;
  
  let mean_coeffs = Tensor.mean cross_sectional_regressions ~dim:[0] ~keepdim:false in
  let std_coeffs = Tensor.std cross_sectional_regressions ~dim:[0] ~keepdim:false ~unbiased:true in
  let t_stats = Tensor.div mean_coeffs (Tensor.div std_coeffs (Tensor.sqrt (Tensor.float (float_of_int n_periods)))) in
  let p_values = p_value t_stats (float_of_int (n_periods - 1)) in
  
  (mean_coeffs, std_coeffs, t_stats, p_values)

let hodrick_correction data lags =
  let n = Tensor.size data 0 in
  let autocovariances = List.init lags (fun lag ->
    let x = Tensor.narrow data ~dim:0 ~start:0 ~length:(n - lag) in
    let y = Tensor.narrow data ~dim:0 ~start:lag ~length:(n - lag) in
    Tensor.mean (Tensor.mul x y) ~dim:[0] ~keepdim:false
  ) in
  let weights = List.init lags (fun lag ->
    1. -. (float_of_int lag) /. (float_of_int lags)
  ) in
  let weighted_sum = List.fold_left2 (fun acc cov w ->
    Tensor.add acc (Tensor.mul cov (Tensor.float w))
  ) (Tensor.float 0.) autocovariances weights in
  Tensor.mul weighted_sum (Tensor.float 2.)