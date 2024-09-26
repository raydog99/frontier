open Torch

let compute_eigendecomposition x =
  Tensor.symeig x ~eigenvectors:true

let compute_overlaps eigenvectors1 eigenvectors2 =
  let overlap_matrix = Tensor.matmul eigenvectors1 (Tensor.transpose eigenvectors2 ~dim0:0 ~dim1:1) in
  Tensor.pow overlap_matrix (Scalar.float 2.0)

let stieltjes_transform x z =
  let n = Tensor.size x ~dim:0 in
  let z_matrix = Tensor.eye n ~dtype:Kind.Float |> Tensor.mul_scalar (Tensor.item z) in
  let resolvent = Tensor.inverse (Tensor.sub z_matrix x) in
  Tensor.mean resolvent

let compute_eigenvector_localization eigenvectors =
  let n = Tensor.size eigenvectors ~dim:0 in
  let squared = Tensor.pow eigenvectors (Scalar.float 2.0) in
  let log_squared = Tensor.log (Tensor.add squared (Tensor.float_scalar 1e-10)) in
  let entropy = Tensor.sum (Tensor.mul squared log_squared) ~dim:0 in
  Utils.safe_div (Tensor.exp entropy) (Tensor.float_scalar (float_of_int n))