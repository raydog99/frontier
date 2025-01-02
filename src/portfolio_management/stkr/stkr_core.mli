open Torch

type model = {
    kernel: Types.kernel;
    transform: Types.transform_fn;
    params: Types.stkr_params;
}

val create : Types.kernel -> Types.transform_fn -> Types.stkr_params -> model
val fit : model -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
val predict : model -> (Tensor.t * Tensor.t) -> Tensor.t -> Tensor.t