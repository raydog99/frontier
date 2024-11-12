open Torch
open Types

val create : config:config -> model
val forward : model -> Tensor.t -> Tensor.t
val loss : model -> Tensor.t -> Tensor.t -> float
val compute_log_likelihood : model -> Tensor.t -> Tensor.t -> float
val train_experts : model -> Tensor.t -> Tensor.t -> model
val train_gates : model -> Tensor.t -> Tensor.t -> model