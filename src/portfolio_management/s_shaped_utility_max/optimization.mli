open Torch

val adam_with_lr_scheduler : Optimizer.t -> float -> int -> unit
val gradient_clipping : Optimizer.t -> float -> unit
val find_optimal_control : Pinn.pinn_params -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val compute_wealth_process : Pinn.pinn_params -> Tensor.t -> (float -> Tensor.t -> Tensor.t) -> int -> Tensor.t list