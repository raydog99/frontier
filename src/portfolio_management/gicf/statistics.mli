open Torch

val mean : float list -> float

val std : float list -> float

val median : float list -> float

val quantile : float list -> float -> float

val compute_correlation : Tensor.t -> Tensor.t -> float

val compute_p_value_normal : float -> float

val compute_p_value_t : float -> int -> float

val compute_p_value_chisq : float -> int -> float