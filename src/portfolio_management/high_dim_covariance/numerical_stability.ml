open Torch

let jitter = 1e-6

let stable_inverse_sqrt matrix =
  let eigenvals, eigenvecs = 
    Tensor.symeig matrix ~eigenvectors:true in
  let min_eigenval = Tensor.min eigenvals |> Tensor.float_value in
  let threshold = Float.max min_eigenval jitter in
  
  let inv_sqrt_vals = Tensor.map eigenvals ~f:(fun x ->
    if x < threshold then 0.0
    else 1.0 /. sqrt x) in
  
  Tensor.mm 
    (Tensor.mm eigenvecs (Tensor.diag inv_sqrt_vals))
    (Tensor.transpose eigenvecs ~dim0:0 ~dim1:1)

let stable_eigendecomposition matrix =
  let stabilized = Tensor.add matrix 
    (Tensor.mul_scalar (Tensor.eye 
      (Tensor.size matrix 0)) jitter) in
  Tensor.symeig stabilized ~eigenvectors:true

let stable_matrix_multiply a b =
  let product = Fast_matrix_ops.rect_multiply a b in
  let sym_part = Tensor.add product 
    (Tensor.transpose product ~dim0:0 ~dim1:1) in
  Tensor.div_scalar sym_part 2.0

let project_to_psd matrix =
  let eigenvals, eigenvecs = stable_eigendecomposition matrix in
  let pos_eigenvals = Tensor.relu eigenvals in
  Tensor.mm 
    (Tensor.mm eigenvecs (Tensor.diag pos_eigenvals))
    (Tensor.transpose eigenvecs ~dim0:0 ~dim1:1)