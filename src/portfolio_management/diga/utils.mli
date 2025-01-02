open Torch

val preprocess_data : Tensor.t -> Tensor.t
val calculate_return : Tensor.t -> Tensor.t
val calculate_amplitude : Tensor.t -> Tensor.t
val calculate_volatility : Tensor.t -> Tensor.t
val evaluate_control_error : Tensor.t -> Tensor.t -> Tensor.t
val evaluate_fidelity : Tensor.t -> Tensor.t -> Tensor.t