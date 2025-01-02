open Torch

let jacobi A =
  let diag = Tensor.diag A in
  Tensor.diag (Tensor.reciprocal diag)

let symmetric_gauss_seidel A =
  let L = Tensor.tril A in
  let U = Tensor.triu A in
  let D = Tensor.diag A in
  let LD = Tensor.(L + D) in
  let LD_inv = Tensor.inverse LD in
  Tensor.(matmul (matmul LD_inv (2. * D - A)) (transpose LD_inv))