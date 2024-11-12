open Torch
open Types

val create : input_dim:int -> expert_type:expert_type -> model
val forward : model -> Tensor.t -> Tensor.t
val compute_mirror_map : model -> Tensor.t -> float
val compute_gate_probability : Tensor.t -> Tensor.t -> Tensor.t
val compute_linear_conditional : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val compute_logistic_conditional : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t