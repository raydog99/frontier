open Torch

type tensor = Tensor.t
type portfolio_weights = tensor
type returns = tensor
type covariance = tensor

type return_type = Simple | Logarithmic

type returns_data = {
  returns: tensor;
  valid_mask: tensor;
  dates: string array;
}

type transaction_costs = {
  fixed_cost: float;
  proportional_cost: float;
  market_impact: float;
}

type constraint_type =
  | LongOnly
  | Turnover of float
  | GroupConstraint of int array array * float array
  | Cardinality of int

type robust_method =
  | ResamplingShrinkage
  | RobustCovariance
  | WorstCase

type distance_metric =
  | Angular
  | Euclidean
  | Correlation
  | CustomMetric of (tensor -> tensor -> float)

type clustering_quality = {
  cophenetic_correlation: float;
  clustering_score: float;
  silhouette_score: float
}

type optimization_result = {
  weights: tensor;
  converged: bool;
  iterations: int;
  objective_value: float
}

type qp_result = {
  solution: tensor;
  converged: bool;
  iterations: int;
  objective_value: float
}

val calculate_returns : ?return_type:return_type -> ?handle_missing:bool -> tensor -> tensor * tensor
val expected_return : portfolio_weights -> returns -> tensor
val portfolio_risk : portfolio_weights -> covariance -> tensor
val rebalance_portfolio : ?costs:transaction_costs option -> portfolio_weights -> portfolio_weights -> tensor -> tensor
val portfolio_risk_with_confidence : portfolio_weights -> covariance -> float -> tensor

val solve : covariance -> portfolio_weights
val solve_long_only : covariance -> portfolio_weights

val solve_robust : covariance -> constraint_type list -> robust_method -> portfolio_weights
val solve_with_cardinality : covariance -> int -> portfolio_weights
val solve_with_turnover : covariance -> portfolio_weights -> float -> optimization_result

val covariance_to_correlation : covariance -> tensor
val correlation_to_distance : tensor -> tensor
val cluster : tensor -> int list array
val quasi_diagonalize : covariance -> int list array -> tensor
val allocate : covariance -> portfolio_weights

val calculate_distance : returns -> distance_metric -> tensor
val adaptive_correlation : returns -> int -> float -> tensor
val evaluate_clustering : tensor -> int list list -> clustering_quality

val naive_estimator : returns -> covariance
val linear_shrinkage : covariance -> covariance
val nonlinear_lp_shrinkage : covariance -> covariance
val nonlinear_stein_shrinkage : covariance -> covariance
val alca_hierarchical : covariance -> covariance
val ycm_estimator : returns -> float -> covariance
val two_step_lp : covariance -> covariance
val two_step_stein : covariance -> covariance
val two_step_ycm : covariance -> covariance

val nested_hierarchical : int -> float -> covariance
val one_factor : int -> float -> float -> covariance
val diagonal_groups : int -> covariance

val solve_qp : ?max_iter:int -> ?tol:float -> tensor -> tensor -> tensor -> tensor -> qp_result

val factor_risk_decomposition : portfolio_weights -> tensor -> tensor -> tensor -> tensor * tensor
val marginal_risk_contributions : portfolio_weights -> covariance -> tensor
val component_risk_contributions : portfolio_weights -> covariance -> tensor

val hhi : portfolio_weights -> tensor
val leverage : portfolio_weights -> tensor
val in_sample_risk : portfolio_weights -> covariance -> tensor
val out_sample_risk : portfolio_weights -> covariance -> covariance -> tensor

val generate_random_returns : n:int -> p:int -> returns
val split_data : tensor -> float -> tensor * tensor
val is_positive_definite : tensor -> bool
val nearest_positive_definite : tensor -> tensor