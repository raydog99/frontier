open Torch
open Matrix_ops

let soft_threshold ~x ~lambda =
  let sign = Tensor.sign x in
  let abs_x = Tensor.abs x in
  let threshold = Tensor.sub abs_x (Tensor.full_like abs_x lambda) in
  let threshold = Tensor.relu threshold in
  Tensor.mul sign threshold

let matrix_l1_prox ~matrix ~lambda =
  let dim = Tensor.size matrix |> List.hd in
  let diag_mask = Tensor.eye dim in
  let off_diag = Tensor.mul (Tensor.sub (Tensor.ones_like diag_mask) diag_mask) matrix in
  let prox_off_diag = soft_threshold ~x:off_diag ~lambda in
  let result = Tensor.add (Tensor.mul diag_mask matrix) prox_off_diag in
  nearest_positive_definite result

let nuclear_prox ~matrix ~lambda =
  let u, s, v = Tensor.svd matrix in
  let s_prox = soft_threshold ~x:s ~lambda in
  let s_diag = Tensor.diag s_prox in
  let vt = Tensor.transpose v ~dim0:0 ~dim1:1 in
  Tensor.mm (Tensor.mm u s_diag) vt

let project_pd_cone ~matrix ~epsilon =
  let eigenvals, eigenvecs = Tensor.symeig matrix ~eigenvectors:true in
  let min_eval = Tensor.min eigenvals |> Tensor.to_float0_exn in
  if min_eval > epsilon then matrix
  else
    let shifted_eigenvals = Tensor.max_pool2d eigenvals epsilon in
    let diag = Tensor.diag shifted_eigenvals in
    let eigenvecs_t = Tensor.transpose eigenvecs ~dim0:0 ~dim1:1 in
    Tensor.mm (Tensor.mm eigenvecs diag) eigenvecs_t