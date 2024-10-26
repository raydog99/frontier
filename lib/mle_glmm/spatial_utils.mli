open Torch

val distance_matrix : Tensor.t -> Tensor.t
(** [distance_matrix locations] computes Euclidean distance matrix *)

val great_circle_distance : Tensor.t -> Tensor.t
(** [great_circle_distance locations] computes great circle distances for lat/lon coordinates *)

val variogram : Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
(** [variogram locations values] computes empirical variogram *)

val local_indicators : Tensor.t -> Tensor.t -> Tensor.t
(** [local_indicators locations values] computes local indicators of spatial association *)

val kriging_weights : Types.model_spec -> Tensor.t -> Tensor.t -> Tensor.t
(** [kriging_weights spec locations values] computes optimal kriging weights *)