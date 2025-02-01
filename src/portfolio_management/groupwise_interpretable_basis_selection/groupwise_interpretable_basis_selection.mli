open Torch

type index_set = int list

type support = {
  indices: int list;
  dimension: int;
}

type cluster = {
  points: Tensor.t;
  prototype: Tensor.t;
  prototype_idx: int;
  radius: float;
  members: index_set;
}

type asset_category =
  | Bond
  | Commodity  
  | Currency
  | DiversifiedPortfolio
  | Equity
  | Alternative
  | Inverse
  | Leveraged
  | RealEstate
  | Volatility

type model_params = {
  lambda: float;
  max_iterations: int;
  convergence_threshold: float;
  correlation_threshold: float;
}

type etf_metadata = {
  symbol: string;
  name: string;
  asset_class: asset_category;
  expense_ratio: float;
  inception_date: string;
  avg_volume: float;
  total_assets: float;
}

type etf_timeseries = {
  dates: string array;
  prices: Tensor.t;
  volumes: Tensor.t;
  returns: Tensor.t;
  adjusted_returns: Tensor.t;
}

type etf_record = {
  metadata: etf_metadata;
  timeseries: etf_timeseries;
}

type regression_result = {
  coefficients: Tensor.t;
  std_errors: Tensor.t;
  t_statistics: Tensor.t;
  r_squared: float;
  residuals: Tensor.t;
}

type significance_matrix = {
  count_matrix: Tensor.t;
  proportion_matrix: Tensor.t;
}

(* Core matrix operations *)
val create_jn : int -> Tensor.t
val create_jn_bar : int -> Tensor.t
val create_identity : ?n:int -> unit -> Tensor.t
val matrix_subset : Tensor.t -> index_set -> Tensor.t
val column_means : Tensor.t -> Tensor.t
val column_vars : Tensor.t -> Tensor.t
val safe_matmul : Tensor.t -> Tensor.t -> Tensor.t

(* Statistical functions *)
val l1_norm : Tensor.t -> float
val l2_norm : Tensor.t -> float
val linf_norm : Tensor.t -> float
val support : Tensor.t -> support
val mean : Tensor.t -> Tensor.t
val var : Tensor.t -> Tensor.t
val std : Tensor.t -> Tensor.t
val correlation_matrix : Tensor.t -> Tensor.t
val ols : Tensor.t -> Tensor.t -> regression_result
val t_test : float -> int -> float
val chi_square_test : float -> int -> float

module Clustering : sig
  val compute_distance : Tensor.t -> Tensor.t -> float
  val compute_max_distance : Tensor.t -> cluster -> float
  val compute_minimax_radius : Tensor.t -> float * int
  val create_cluster : Tensor.t -> cluster
  val merge_clusters : cluster -> cluster -> cluster
  val cluster : Tensor.t -> model_params -> cluster list
end

module Lasso : sig
  val default_params : model_params
  val fit_lasso : ?params:model_params -> Tensor.t -> Tensor.t -> Tensor.t
  val select_lambda : Tensor.t -> Tensor.t -> int -> float
end

module Gibs : sig
  type gibs_config = {
    correlation_threshold: float;
    min_observations: int;
    max_basis_assets: int;
    lambda_multiplier: float;
    include_ff5: bool;
  }

  val transform_basis_assets : Tensor.t -> etf_record array -> Tensor.t array
  val select_category_basis : etf_record array -> gibs_config -> etf_record list
  val run : etf_record array -> etf_record array -> gibs_config -> 
    Tensor.t array * int list array * Tensor.t * Tensor.t
end

module SignificanceMatrices : sig
  val compute_basis_counts : 
    asset_category list -> asset_category list -> int list array -> Tensor.t
  val compute_proportions : Tensor.t -> Tensor.t
end

module FactorAnalysis : sig
  type factor_config = {
    min_variance_explained: float;
    max_correlation: float;
    significance_level: float;
    stability_threshold: float;
  }

  val compute_factor_exposures : 
    Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t
  val analyze_stability : Tensor.t -> Tensor.t -> int -> Tensor.t
  val select_factors : Tensor.t -> Tensor.t -> factor_config -> int list
  val orthogonalize_factors : Tensor.t -> Tensor.t
end