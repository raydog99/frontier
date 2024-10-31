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

let default = {
  device = if Cuda.is_available () then GPU 0 else CPU;
  memory_mode = Standard;
  num_threads = 4;
  batch_size = 1000;
  sparsity_threshold = 1e-6;
  use_mixed_precision = true;
}

let get_device_tensor config t =
  match config.device with
  | CPU -> t
  | GPU device_id -> Tensor.cuda t ~device_id