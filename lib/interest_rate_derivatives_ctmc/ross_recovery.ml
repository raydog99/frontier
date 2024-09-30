open Torch
open Ctmc

let recover_real_world_dynamics ctmc =
  let g_minus_r = g_minus_r ctmc in
  let eigenvalues, eigenvectors = Tensor.symeig g_minus_r ~eigenvectors:true in
  let min_eigenvalue = Tensor.min eigenvalues in
  let min_eigenvector_index = Tensor.argmin eigenvalues ~dim:0 ~keepdim:false in
  let min_eigenvector = Tensor.select eigenvectors 1 (Tensor.int_repr min_eigenvector_index) in
  let pi = Tensor.abs min_eigenvector in
  let pi_diag = Tensor.diag pi in
  let pi_diag_inv = Tensor.diag (Tensor.reciprocal pi) in
  let real_world_generator = Tensor.matmul (Tensor.matmul pi_diag ctmc.generator) pi_diag_inv in
  create real_world_generator ctmc.rates