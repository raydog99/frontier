open Torch
open Types

type kpca_result = {
  eigenvalues: Tensor.t;
  eigenvectors: Tensor.t;
  mean: Tensor.t;
}

let rbf_kernel sigma x y =
  let diff = Tensor.(x - y) in
  Tensor.(exp (neg (sum (pow diff (Scalar.f 2.)) ~dim:[1]) / (Scalar.f 2. * Scalar.f (sigma ** 2.))))

let compute_gram_matrix sigma data =
  let n = Tensor.shape data |> List.hd in
  let gram = Tensor.zeros [n; n] in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      let k = rbf_kernel sigma (Tensor.select data ~dim:0 ~index:i) (Tensor.select data ~dim:0 ~index:j) in
      Tensor.set gram [i; j] k
    done
  done;
  gram

let kernel_pca sigma data n_components =
  let gram = compute_gram_matrix sigma data in
  let n = Tensor.shape data |> List.hd in
  let mean = Tensor.mean data ~dim:[0] ~keepdim:true in
  let centered_gram = Tensor.(gram - (f 1. / f (float n)) * (mm gram (ones [n; n])) - 
                              (mm (ones [n; n]) gram) + 
                              (f 1. / f (float (n * n))) * (mm (mm (ones [n; n]) gram) (ones [n; n]))) in
  let eigenvalues, eigenvectors = Tensor.symeig centered_gram ~eigenvectors:true in
  let sorted_indices = Tensor.argsort eigenvalues ~descending:true in
  let top_k_indices = Tensor.slice sorted_indices ~dim:0 ~start:None ~end:(Some n_components) ~step:None in
  let top_eigenvalues = Tensor.index_select eigenvalues ~dim:0 ~index:top_k_indices in
  let top_eigenvectors = Tensor.index_select eigenvectors ~dim:1 ~index:top_k_indices in
  { eigenvalues = top_eigenvalues; eigenvectors = top_eigenvectors; mean }

let project_data kpca_result data =
  let centered_data = Tensor.(data - kpca_result.mean) in
  Tensor.mm centered_data kpca_result.eigenvectors

let reconstruct_data kpca_result projected_data =
  Tensor.(mm projected_data (transpose kpca_result.eigenvectors ~dim0:0 ~dim1:1) + kpca_result.mean)

let functional_regression reference_yields response_yields n_components sigma =
  let kpca_result = kernel_pca sigma reference_yields n_components in
  let projected_reference = project_data kpca_result reference_yields in
  let beta = Tensor.(mm (pinverse projected_reference) response_yields) in
  (kpca_result, beta)

let predict_functional_regression kpca_result beta new_reference_yields =
  let projected_new = project_data kpca_result new_reference_yields in
  Tensor.mm projected_new beta