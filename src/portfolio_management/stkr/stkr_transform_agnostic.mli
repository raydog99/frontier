open Torch

val create_stable_inverse_laplacian : float -> float -> Types.transform_fn

module StableKernelPCA : sig
    type pca_state = {
      eigenvectors: Tensor.t;
      eigenvalues: Tensor.t;
      explained_variance_ratio: Tensor.t;
    }

    val fit : Types.kernel -> int -> float -> Tensor.t -> pca_state
    val transform : pca_state -> Types.kernel -> Tensor.t -> Tensor.t -> Tensor.t
end