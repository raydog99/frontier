open Torch

val sparse_projection : 
  Tensor.t -> Tensor.t -> float -> Tensor.t
(** [sparse_projection x theta lambda] computes the sparse projection map 
    θ* = argmin{n⁻¹||Xθ - Xu||² + λ||u||₁} *)

val sparse_projection_distributed : 
  Type.distributed_data -> Tensor.t -> float -> Tensor.t
(** [sparse_projection_distributed data theta lambda] computes sparse projection 
    using distributed sufficient statistics *)