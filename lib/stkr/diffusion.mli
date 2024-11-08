open Torch

type diffusion_kernel = {
    base_kernel: Types.kernel;
    time_steps: float;
    eigenvalues: Tensor.t option;
    eigenfunctions: Tensor.t option;
}

val diffusion_distance : Types.kernel -> float -> Tensor.t -> Tensor.t -> float
val smoothness_measure : Types.kernel -> float -> Tensor.t -> Tensor.t -> Tensor.t