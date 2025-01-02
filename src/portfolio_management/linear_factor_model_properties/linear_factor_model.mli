open Torch

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

module Utils : sig
  val pseudoinverse : Tensor.t -> Tensor.t
  val is_in_image : Tensor.t -> Tensor.t -> bool
  val safe_division : Tensor.t -> Tensor.t -> Tensor.t
  val matrix_rank : Tensor.t -> Tensor.t
end

module RiskPremium : sig
  val is_residual_risk_unpriced : model_params -> bool
  val check_risk_premium_condition : model_params -> bool
  val absence_of_arbitrage : model_params -> bool
  val mean_variance_efficient_portfolio : model_params -> Tensor.t
  val stochastic_discount_factor : model_params -> (Tensor.t -> Tensor.t)
  val calculate_risk_premia : model_params -> Tensor.t
end

module SharpeRatio : sig
  val calculate_sharpe_ratio : Tensor.t -> Tensor.t -> Tensor.t
  val compare_sharpe_ratios : model_params -> bool
  val check_spanning_condition : model_params -> bool
  val maximum_sharpe_ratio : model_params -> Tensor.t
end

module Statistics : sig
  val calculate_t_statistics : model_params -> Tensor.t
  val calculate_r_squared : model_params -> Tensor.t
  val perform_f_test : model_params -> Tensor.t
  val calculate_information_ratio : model_params -> Tensor.t
end

module TimeVarying : sig
  type time_varying_params = {
    returns : Tensor.t;
    characteristics : Tensor.t;
    factors : Tensor.t;
    residuals : Tensor.t;
    weight_matrix : Tensor.t;
  }

  val create_time_varying_params :
    returns:Tensor.t -> characteristics:Tensor.t -> factors:Tensor.t ->
    residuals:Tensor.t -> weight_matrix:Tensor.t -> time_varying_params

  val calculate_time_varying_returns : time_varying_params -> Tensor.t
end

module Optimization : sig
  val optimize_weights : model_params -> float -> int -> Tensor.t
  val cross_validate : model_params -> int -> float
end

module Visualization : sig
  val plot_returns_vs_predicted : model_params -> Tensor.t -> unit
  val plot_factor_contributions : model_params -> unit
  val plot_correlation_matrix : model_params -> unit
end

val calculate_returns : model_params -> Tensor.t
val calculate_tradable_factors : model_params -> Tensor.t
val decompose_covariance : model_params -> Tensor.t
val factors_residuals_covariance : model_params -> Tensor.t
val check_covariance_equality : model_params -> bool
val zero_matrix_product : model_params -> Tensor.t
val check_full_rank_assumption : model_params -> bool
val factor_and_residual_correlation_covariance : model_params -> bool
val zero_matrix_product_relation : model_params -> bool
val characteristics_as_covariances : model_params -> bool
val factor_mve_portfolio_spanning : model_params -> bool
val spanning_and_uncorrelated_factors_residuals : model_params -> bool
val gls_factors_spanning : model_params -> bool

val create_model_params :
  returns:returns -> characteristics:characteristics -> factors:factors ->
  residuals:residuals -> weight_matrix:weight_matrix -> returns_mean:mean ->
  returns_cov:covariance -> factors_mean:mean -> factors_cov:covariance ->
  residuals_cov:covariance -> model_params

val conditional_linear_factor_model :
  model_params ->
  ((Tensor.t * Tensor.t * Tensor.t * Tensor.t * bool * Tensor.t) *
   (bool * bool * bool * Tensor.t) *
   (bool * bool) *
   (bool * bool * bool * bool * bool * bool), string) result

val benchmark_model : model_params -> unit