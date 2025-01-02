open Torch

type t
type model_type = Unrestricted | Restricted of int | FAVAR of int
type frequency = Daily | Weekly | Monthly | Quarterly
type observation = {
  value : float;
  frequency : frequency;
  date : float;
}

val create : int -> int -> int -> Tensor.t -> model_type -> t
val handle_missing_data : Tensor.t -> Tensor.t
val fit : t -> float -> t
val forecast : t -> int -> Tensor.t
val compute_explained_variance : t -> float
val information_criterion : t -> int -> float list
val select_number_of_factors : t -> int -> int
val bootstrap_confidence_intervals : t -> int -> float -> Tensor.t * Tensor.t
val detect_structural_breaks : t -> int -> Tensor.t
val summary_statistics : t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t