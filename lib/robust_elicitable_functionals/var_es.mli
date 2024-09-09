open Torch
open Elicitable_functional

val create : float -> Elicitable_functional.t
val evaluate : Elicitable_functional.t -> Torch.Tensor.t -> Torch.Tensor.t