open Torch

val run_wealth_process : Model.model_params -> Tensor.t -> (float -> Tensor.t -> Tensor.t) -> int -> int -> Tensor.t list list
val compute_expected_utility : Utility.Utility -> Tensor.t list list -> Tensor.t