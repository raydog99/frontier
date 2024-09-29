open Torch

val tensor_to_float : Tensor.t -> float
val correlation : Tensor.t -> Tensor.t -> float
val t_statistic : Tensor.t -> Tensor.t -> int -> Tensor.t
val p_value : Tensor.t -> float -> Tensor.t
val calculate_irr : Tensor.t -> Tensor.t
val newey_west_adjustment : Tensor.t -> int -> Tensor.t
val bootstrap_confidence_interval : Tensor.t -> float -> int -> Tensor.t * Tensor.t
val rolling_window : Tensor.t -> int -> int -> (Tensor.t -> 'a) -> 'a list
val hansen_jagannathan_distance : Tensor.t -> Tensor.t -> Tensor.t
val gmm_estimation : Tensor.t -> Tensor.t -> int -> Tensor.t
val calculate_turnover : Factor.t -> Tensor.t
val calculate_information_coefficient : Factor.t -> Tensor.t -> float * Tensor.t * Tensor.t
val calculate_autocorrelation : Factor.t -> int -> float
val fama_macbeth_regression : Tensor.t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t
val hodrick_correction : Tensor.t -> int -> Tensor.t