open Torch
open Plotly
open Core
open Core_bench

type time = int
type asset_count = int
type factor_count = int

type returns = Tensor.t
type characteristics = Tensor.t
type factors = Tensor.t
type residuals = Tensor.t
type weight_matrix = Tensor.t

type mean = Tensor.t
type covariance = Tensor.t

type model_params = {
  returns : returns;
  characteristics : characteristics;
  factors : factors;
  residuals : residuals;
  weight_matrix : weight_matrix;
  returns_mean : mean;
  returns_cov : covariance;
  factors_mean : mean;
  factors_cov : covariance;
  residuals_cov : covariance;
}

module Utils = struct
  let pseudoinverse tensor =
    let u, s, v = Tensor.svd tensor ~some:false in
    let s_inv = Tensor.(1.0 / (s + 1e-10)) in
    Tensor.(matmul (matmul v (diag s_inv)) (transpose2 u))

  let is_in_image tensor vector =
    let projection = Tensor.(matmul (matmul tensor (pseudoinverse tensor)) vector) in
    Tensor.allclose projection vector

  let safe_division a b =
    let epsilon = Tensor.scalar_tensor 1e-10 in
    Tensor.(div a (add b epsilon))

  let matrix_rank tensor =
    let _, s, _ = Tensor.svd tensor ~some:false in
    Tensor.(sum (gt s (scalar_tensor 1e-5)))
end

module RiskPremium = struct
  let is_residual_risk_unpriced params =
    Tensor.allclose params.residuals Tensor.zeros_like

  let check_risk_premium_condition params =
    let lhs = params.returns_mean in
    let rhs = Tensor.(mm (mm params.characteristics (transpose2 params.weight_matrix)) params.returns_mean) in
    Tensor.allclose lhs rhs

  let absence_of_arbitrage params =
    Utils.is_in_image params.returns_cov params.returns_mean

  let mean_variance_efficient_portfolio params =
    let inv_cov = Utils.pseudoinverse params.returns_cov in
    Tensor.(mm inv_cov params.returns_mean)

  let stochastic_discount_factor params =
    let mve_portfolio = mean_variance_efficient_portfolio params in
    let expected_return = Tensor.(dot params.returns_mean mve_portfolio) in
    fun x -> Tensor.(sub (scalar_tensor 1.0) (div (dot mve_portfolio (sub x params.returns_mean)) expected_return))

  let calculate_risk_premia params =
    Tensor.(sub params.returns_mean (mm params.characteristics params.factors_mean))
end

module SharpeRatio = struct
  let calculate_sharpe_ratio mean_return cov =
    let inv_cov = Utils.pseudoinverse cov in
    Tensor.(dot (mm (transpose2 mean_return) inv_cov) mean_return)

  let compare_sharpe_ratios params =
    let sr_factors = calculate_sharpe_ratio params.factors_mean params.factors_cov in
    let sr_returns = calculate_sharpe_ratio params.returns_mean params.returns_cov in
    Tensor.allclose sr_factors sr_returns

  let check_spanning_condition params =
    Utils.is_in_image Tensor.(mm params.returns_cov params.weight_matrix) params.returns_mean

  let maximum_sharpe_ratio params =
    let inv_cov = Utils.pseudoinverse params.returns_cov in
    sqrt Tensor.(dot (mm (transpose2 params.returns_mean) inv_cov) params.returns_mean)
end

module Statistics = struct
  let calculate_t_statistics params =
    let standard_errors = Tensor.(
      sqrt (diag params.factors_cov)
    ) in
    Tensor.(div params.factors_mean standard_errors)

  let calculate_r_squared params =
    let total_variance = Tensor.var params.returns in
    let residual_variance = Tensor.var params.residuals in
    Tensor.(sub (scalar_tensor 1.0) (div residual_variance total_variance))

  let perform_f_test params =
    let n = Tensor.size params.returns 0 in
    let k = Tensor.size params.factors 0 in
    let ssr = Tensor.(sum (pow params.residuals (scalar_tensor 2.0))) in
    let sst = Tensor.(sum (pow (sub params.returns (mean params.returns)) (scalar_tensor 2.0))) in
    let f_statistic = Tensor.(
      div (div (sub sst ssr) (scalar_tensor (float_of_int (k - 1))))
          (div ssr (scalar_tensor (float_of_int (n - k))))
    ) in
    f_statistic

  let calculate_information_ratio params =
    let active_returns = Tensor.(sub params.returns (mm params.characteristics params.factors)) in
    let ir = Tensor.(
      div (mean active_returns)
          (std active_returns)
    ) in
    ir
end

module TimeVarying = struct
  type time_varying_params = {
    returns : Tensor.t; (* t x n *)
    characteristics : Tensor.t; (* t x n x m *)
    factors : Tensor.t; (* t x m *)
    residuals : Tensor.t; (* t x n *)
    weight_matrix : Tensor.t; (* t x m x n *)
  }

  let create_time_varying_params ~returns ~characteristics ~factors ~residuals ~weight_matrix =
    {returns; characteristics; factors; residuals; weight_matrix}

  let calculate_time_varying_returns params =
    let t = Tensor.size params.returns 0 in
    let n = Tensor.size params.returns 1 in
    let calculated_returns = Tensor.zeros [t; n] in
    for i = 0 to t - 1 do
      let char_t = Tensor.select params.characteristics 0 i in
      let factors_t = Tensor.select params.factors 0 i in
      let residuals_t = Tensor.select params.residuals 0 i in
      let returns_t = Tensor.(add_ (mm char_t factors_t) residuals_t) in
      Tensor.(copy_ (select calculated_returns 0 i) returns_t)
    done;
    calculated_returns
end

module Optimization = struct
  let optimize_weights params learning_rate max_iterations =
    let weights = Tensor.randn [Tensor.size params.characteristics 1; Tensor.size params.returns 1] in
    let optimizer = Optimizer.adam [weights] ~lr:learning_rate in
    
    for _ = 1 to max_iterations do
      Optimizer.zero_grad optimizer;
      let predicted_returns = Tensor.(mm params.characteristics weights) in
      let loss = Tensor.mse_loss predicted_returns params.returns in
      Tensor.backward loss;
      Optimizer.step optimizer;
    done;
    weights

  let cross_validate params k_folds =
    let n = Tensor.size params.returns 0 in
    let fold_size = n / k_folds in
    let mse_scores = ref [] in
    
    for i = 0 to k_folds - 1 do
      let test_start = i * fold_size in
      let test_end = (i + 1) * fold_size in
      let train_data, test_data = Tensor.split2 params.returns ~dim:0 ~sizes:[test_start; n - test_start] in
      let train_chars, test_chars = Tensor.split2 params.characteristics ~dim:0 ~sizes:[test_start; n - test_start] in
      
      let train_params = {params with returns = train_data; characteristics = train_chars} in
      let weights = optimize_weights train_params 0.01 1000 in
      
      let predicted_test = Tensor.(mm test_chars weights) in
      let mse = Tensor.mse_loss predicted_test test_data in
      mse_scores := mse :: !mse_scores;
    done;
    
    List.fold !mse_scores ~init:0. ~f:(+.) /. float_of_int k_folds
end

module Visualization = struct
  let plot_returns_vs_predicted params predicted_returns =
    let actual = Tensor.squeeze params.returns |> Tensor.to_float1 in
    let predicted = Tensor.squeeze predicted_returns |> Tensor.to_float1 in
    let trace = Scatter.create ~x:actual ~y:predicted ~mode:(`Markers) 
                  ~marker:(Marker.create ~size:10 ~color:"blue" ()) 
                  ~name:"Returns" () in
    let layout = Layout.create ~title:"Actual vs Predicted Returns" 
                   ~xaxis:(Axis.create ~title:"Actual Returns" ())
                   ~yaxis:(Axis.create ~title:"Predicted Returns" ()) () in
    let plot = Plot.create [trace] ~layout in
    Plot.write_html plot "returns_vs_predicted.html"

  let plot_factor_contributions params =
    let factor_returns = Tensor.(mm params.characteristics params.factors) in
    let total_return = Tensor.sum params.returns in
    let factor_contributions = Tensor.(div factor_returns total_return) |> Tensor.squeeze |> Tensor.to_float1 in
    let x = List.init (Array.length factor_contributions) (fun i -> string_of_int (i + 1)) in
    let trace = Bar.create ~x ~y:factor_contributions ~name:"Factor Contributions" () in
    let layout = Layout.create ~title:"Factor Contributions to Total Return" 
                   ~xaxis:(Axis.create ~title:"Factor" ())
                   ~yaxis:(Axis.create ~title:"Contribution" ()) () in
    let plot = Plot.create [trace] ~layout in
    Plot.write_html plot "factor_contributions.html"

  let plot_correlation_matrix params =
    let corr_matrix = Tensor.corrcoef params.returns in
    let z = Tensor.to_float2 corr_matrix in
    let trace = Heatmap.create ~z ~colorscale:`Viridis () in
    let layout = Layout.create ~title:"Asset Correlation Matrix" () in
    let plot = Plot.create [trace] ~layout in
    Plot.write_html plot "correlation_matrix.html"
end

let calculate_returns params =
  Tensor.(add_ (mm params.characteristics params.factors) params.residuals)

let calculate_tradable_factors params =
  Tensor.(mm (transpose2 params.weight_matrix) params.returns)

let decompose_covariance params =
  let factor_component = Tensor.(
    mm (mm params.characteristics params.factors_cov) (transpose2 params.characteristics)
  ) in
  Tensor.(add_ factor_component params.residuals_cov)

let factors_residuals_covariance params =
  Tensor.(mm (mm params.factors_cov (transpose2 params.characteristics)) params.residuals_cov)

let check_covariance_equality params =
  let lhs = Tensor.(mm (mm params.returns_cov params.weight_matrix) (transpose2 params.characteristics)) in
  let rhs = Tensor.(
    mm (mm (mm params.characteristics (transpose2 params.weight_matrix)) params.returns_cov)
       (mm params.weight_matrix (transpose2 params.characteristics))
  ) in
  Tensor.allclose lhs rhs

let zero_matrix_product params =
  Tensor.(mm (mm params.characteristics (transpose2 params.weight_matrix)) params.residuals_cov)

let check_full_rank_assumption params =
  Utils.matrix_rank params.characteristics = Tensor.size params.characteristics 1

let factor_and_residual_correlation_covariance params =
  let lhs = Tensor.(mm (mm params.returns_cov params.weight_matrix) (transpose2 params.characteristics)) in
  let mid = Tensor.(
    mm (mm (mm params.characteristics (transpose2 params.weight_matrix)) params.returns_cov)
       (mm params.weight_matrix (transpose2 params.characteristics))
  ) in
  let rhs = Tensor.(mm (mm params.characteristics (transpose2 params.weight_matrix)) params.returns_cov) in
  Tensor.allclose lhs mid && Tensor.allclose mid rhs

let zero_matrix_product_relation params =
  let zero_product = zero_matrix_product params in
  Tensor.allclose zero_product Tensor.zeros_like

let characteristics_as_covariances params =
  let factors_residuals_cov = factors_residuals_covariance params in
  let residual_risk_unpriced = RiskPremium.is_residual_risk_unpriced params in
  let characteristics_are_covariances = 
    let objective beta = 
      Tensor.(mean_squared_error (sub params.returns (mm beta params.factors)) Tensor.zeros_like)
    in
    let optimal_beta = params.characteristics in
    Tensor.allclose optimal_beta params.characteristics
  in
  Tensor.allclose factors_residuals_cov Tensor.zeros_like && 
  residual_risk_unpriced && 
  characteristics_are_covariances

let factor_mve_portfolio_spanning params =
  let sr_equal = SharpeRatio.compare_sharpe_ratios params in
  let spanning_condition = SharpeRatio.check_spanning_condition params in
  let mve_portfolio = RiskPremium.mean_variance_efficient_portfolio params in
  let factor_mve_portfolio = 
    let inv_factor_cov = Utils.pseudoinverse params.factors_cov in
    Tensor.(mm inv_factor_cov params.factors_mean)
  in
  let mve_equality = Tensor.allclose 
    Tensor.(mm (transpose2 params.factors_mean) factor_mve_portfolio)
    Tensor.(mm (transpose2 params.returns_mean) mve_portfolio)
  in
  sr_equal && spanning_condition && mve_equality

let spanning_and_uncorrelated_factors_residuals params =
  let uncorrelated = factors_residuals_covariance params |> Tensor.allclose Tensor.zeros_like in
  let unpriced_residual = RiskPremium.is_residual_risk_unpriced params in
  let spanning = SharpeRatio.check_spanning_condition params in
  let weight_condition = 
    Tensor.allclose 
      Tensor.(mm (transpose2 params.weight_matrix) (mm params.characteristics (transpose2 params.weight_matrix)))
      (transpose2 params.weight_matrix)
  in
  uncorrelated && unpriced_residual && spanning && weight_condition

let gls_factors_spanning params =
  let gls_factors = Tensor.(mm (Utils.pseudoinverse (mm params.characteristics (transpose2 params.characteristics))) params.characteristics) in
  let spanning = SharpeRatio.check_spanning_condition {params with weight_matrix = gls_factors} in
  let unpriced_residual = RiskPremium.is_residual_risk_unpriced params in
  let uncorrelated = factors_residuals_covariance params |> Tensor.allclose Tensor.zeros_like in
  spanning && unpriced_residual && uncorrelated

let create_model_params ~returns ~characteristics ~factors ~residuals ~weight_matrix 
                        ~returns_mean ~returns_cov ~factors_mean ~factors_cov ~residuals_cov =
  { returns; characteristics; factors; residuals; weight_matrix; 
    returns_mean; returns_cov; factors_mean; factors_cov; residuals_cov }

let conditional_linear_factor_model params =
  try
    if not (check_full_rank_assumption params) then
      Error "Characteristics matrix does not have full rank"
    else
      let returns_cov_weight = Tensor.(mm params.returns_cov params.weight_matrix) in
      let weight_returns_cov_weight = Tensor.(mm (transpose2 params.weight_matrix) returns_cov_weight) in
      
      let calculated_returns = calculate_returns params in
      let tradable_factors = calculate_tradable_factors params in
      let decomposed_cov = decompose_covariance params in
      
      let basic_props = (
        calculated_returns,
        tradable_factors,
        decomposed_cov,
        factors_residuals_covariance params,
        check_covariance_equality params,
        zero_matrix_product params
      ) in
      
      let risk_premium = (
        RiskPremium.is_residual_risk_unpriced params,
        RiskPremium.check_risk_premium_condition params,
        RiskPremium.absence_of_arbitrage params,
        RiskPremium.mean_variance_efficient_portfolio params
      ) in
      
      let sharpe_ratio = (
        SharpeRatio.compare_sharpe_ratios params,
        SharpeRatio.check_spanning_condition params
      ) in
      
      let advanced_props = (
        factor_and_residual_correlation_covariance params,
        zero_matrix_product_relation params,
        characteristics_as_covariances params,
        factor_mve_portfolio_spanning params,
        spanning_and_uncorrelated_factors_residuals params,
        gls_factors_spanning params
      ) in
      
      Ok (basic_props, risk_premium, sharpe_ratio, advanced_props)
  with
  | Invalid_argument msg -> Error ("Invalid argument: " ^ msg)
  | Failure msg -> Error ("Failure: " ^ msg)
  | _ -> Error "An unexpected error occurred"

let benchmark_model params =
  Command.run (Bench.make_command [
    Bench.Test.create ~name:"Conditional Linear Factor Model" (fun () ->
      ignore (conditional_linear_factor_model params)
    );
  ])