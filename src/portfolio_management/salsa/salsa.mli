open Torch
open Types

val compute_leverage_scores : Tensor.t -> sample_size -> sample_size -> Tensor.t
val compute_scaling_factors : Tensor.t -> int -> Tensor.t

val estimate_ar_params : Tensor.t -> int -> Tensor.t
val estimate_white_noise : Tensor.t -> int -> Tensor.t * Tensor.t
val fit_arma : Tensor.t -> int -> int -> int -> arma_params
val compute_diagnostics : arma_params -> Tensor.t -> diagnostic_stats
val parallel_select_order : Tensor.t -> int -> int -> int -> (int * int * arma_params) option
val make_stationary : Tensor.t -> int -> Tensor.t * int