open Torch

type t
type model_type = Static | Dynamic of int | FAVAR of int
type frequency = Daily | Weekly | Monthly | Quarterly
type observation = {
  value : float;
  frequency : frequency;
  date : float;
}

val create : int -> int -> int -> model_type -> t
val create_favar : int -> int -> int -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val estimate_factors_pca : t -> Tensor.t -> Tensor.t
val estimate_loadings_ols : t -> Tensor.t -> Tensor.t -> Tensor.t
val fit : t -> Tensor.t -> int -> [`PCA | `QMLE of float | `FAVAR of float] -> t
val forecast_var : t -> Tensor.t -> int -> Tensor.t
val handle_missing_data : Tensor.t -> Tensor.t
val bootstrap_confidence_intervals : t -> Tensor.t -> int -> float -> Tensor.t * Tensor.t
val compute_factor_contributions : t -> Tensor.t -> Tensor.t
val plot_factor_loadings : t -> unit
val plot_factors : t -> Tensor.t -> unit
val diagnostic_tests : t -> Tensor.t -> float array * float * float array
val interpolate_mixed_frequency : observation list -> (float -> float)
val impulse_response : t -> int -> float array -> float array array
val forecast_evaluation : t -> Tensor.t -> int -> float * float
val hannan_quinn_criterion : t -> Tensor.t -> float
val select_optimal_factors_hq : Tensor.t -> int -> int
val detect_structural_breaks : t -> Tensor.t -> int -> Tensor.t
val regularized_fit : t -> Tensor.t -> int -> float -> t
val summary_statistics : t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t * Tensor.t
val plot_factor_heatmap : t -> Tensor.t -> unit