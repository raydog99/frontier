open Torch

type kpca_result = {
  eigenvalues: Tensor.t;
  eigenvectors: Tensor.t;
  mean: Tensor.t;
}

val rbf_kernel : float -> Tensor.t -> Tensor.t -> Tensor.t

val compute_gram_matrix : float -> Tensor.t -> Tensor.t

val kernel_pca : float -> Tensor.t -> int -> kpca_result

val project_data : kpca_result -> Tensor.t -> Tensor.t

val reconstruct_data : kpca_result -> Tensor.t -> Tensor.t

val functional_regression : Tensor.t -> Tensor.t -> int -> float -> kpca_result * Tensor.t

val predict_functional_regression : kpca_result -> Tensor.t -> Tensor.t -> Tensor.t