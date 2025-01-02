open Torch

type device = CPU | GPU of int
type memory_mode = Standard | LowMemory | Distributed

type t = {
    device: device;
    memory_mode: memory_mode;
    num_threads: int;
    batch_size: int;
    sparsity_threshold: float;
    use_mixed_precision: bool;
}

val default : t
val get_device_tensor : t -> Tensor.t -> Tensor.t