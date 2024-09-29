open Torch

type t = {
  name: string;
  factors: Factor.t list;
}

val create : string -> Factor.t list -> t
val get_factor_by_name : t -> string -> Factor.t option
val factor_correlations : t -> Tensor.t
val spanning_regression : Factor.t -> Factor.t list -> Tensor.t * Tensor.t * Tensor.t * Tensor.t
val grs_test : t -> Factor.t list -> Tensor.t * Tensor.t
val compare_models : t -> t -> Config.t -> (string * bool * bool * Tensor.t * Tensor.t * Tensor.t * Tensor.t * Tensor.t * Tensor.t * Tensor.t) list
val analyze_premiums_over_time : t -> Config.t -> (string * (int * (Tensor.t * Tensor.t * Tensor.t * Tensor.t)) list) list
val calculate_factor_loadings : t -> Tensor.t -> Tensor.t
val calculate_information_ratio : t -> Tensor.t -> Tensor.t
val calculate_tracking_error : t -> Tensor.t -> Tensor.t
val cross_sectional_regression : t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t
val rolling_factor_loadings : t -> Tensor.t -> Config.t -> Tensor.t list
val calculate_factor_exposures : t -> Tensor.t -> Tensor.t
val perform_spanning_tests : t -> t -> (Tensor.t * Tensor.t * Tensor.t * Tensor.t) * (Tensor.t * Tensor.t * Tensor.t * Tensor.t)
val calculate_factor_exposures_timeseries : t -> Tensor.t -> int -> Tensor.t
val calculate_cross_sectional_regression : t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t