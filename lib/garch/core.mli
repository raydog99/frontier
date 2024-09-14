open Torch
open Types

val estimate_variance_cc : Tensor.t -> Tensor.t
val estimate_variance_p : Tensor.t -> Tensor.t -> Tensor.t
val estimate_variance_rs : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val estimate_variance_o : Tensor.t -> Tensor.t
val estimate_variance_c : Tensor.t -> Tensor.t
val calculate_k0 : int -> float -> float
val estimate_variance : ?alpha:float -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val estimate_garch_parameters_generic : garch_model -> Tensor.t -> max_iter:int -> learning_rate:float -> float * float * float * float
val forecast_volatility : garch_model -> float * float * float * float -> Tensor.t -> int -> Tensor.t
val calculate_confidence_interval : Tensor.t -> float -> Tensor.t * Tensor.t
val calculate_var : Tensor.t -> float -> Tensor.t
val calculate_es : Tensor.t -> float -> Tensor.t