open Torch
open Types

val compute : data -> model_params -> Tensor.t
val compute_expected : data -> model_params -> Tensor.t
val observed_information : data -> model_params -> Tensor.t