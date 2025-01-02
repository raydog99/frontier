open Torch

type optimized_model = {
    base_model: Stkr_core.model;
    kernel_cache: Kernel_cache.t;
    chunk_size: int;
    use_parallel: bool;
}

val create : Stkr_core.model -> int -> int -> bool -> optimized_model
val fit : optimized_model -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
val predict : optimized_model -> (Tensor.t * Tensor.t) -> Tensor.t -> Tensor.t