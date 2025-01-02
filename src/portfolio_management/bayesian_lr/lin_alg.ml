open Torch

let gram_matrix x =
  let xt = transpose2D x ~dim0:0 ~dim1:1 in
  matmul xt x

let cross_product x y =
  let xt = transpose2D x ~dim0:0 ~dim1:1 in
  matmul xt y

let solve a b =
  let u, s, v = svd a in
  let s_inv = div_scalar (ones_like s) s in
  let ut = transpose2D u ~dim0:0 ~dim1:1 in
  let v = transpose2D v ~dim0:0 ~dim1:1 in
  matmul (matmul (matmul v (diag s_inv)) ut) b