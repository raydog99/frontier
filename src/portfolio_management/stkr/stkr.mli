open Torch

type model = [
    | `Base of Stkr_core.model
    | `Optimized of OptimizedSTKR.optimized_model
]

val create : Types.kernel -> Types.transform_fn -> Types.stkr_params -> model
val create_optimized : Types.kernel -> Types.transform_fn -> Types.stkr_params -> 
    int -> int -> bool -> model
val smart_fit : model -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
val smart_predict : model -> (Tensor.t * Tensor.t) -> Tensor.t -> Tensor.t