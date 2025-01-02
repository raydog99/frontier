open Torch

exception Invalid_Input of string
exception Convergence_Error of string
exception Numerical_Error of string

val validate_input : Tensor.t -> unit
val check_convergence : float -> float -> unit
val safe_division : Tensor.t -> Tensor.t -> Tensor.t