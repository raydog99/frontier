open Torch
open Types

let mean samples =
  Tensor.mean samples ~dim:[0] ~keepdim:false

let covariance samples sample_mean =
  let n = Tensor.size samples 0 |> float_of_int in
  let centered = Tensor.sub samples (Tensor.expand_as sample_mean samples) in
  let cov = Tensor.mm (Tensor.transpose centered 0 1) centered in
  Tensor.div_scalar cov n

let spectral_gap transition_matrix =
  let eigenvals = Tensor.linalg_eigvals transition_matrix in
  let sorted_eigs = Tensor.sort eigenvals ~descending:true |> fst in
  let lambda1 = Tensor.get sorted_eigs 0 |> Tensor.item in
  let lambda2 = Tensor.get sorted_eigs 1 |> Tensor.item in
  1.0 -. (abs_float lambda2) /. lambda1

let estimate_trace m n_samples =
  let d = Tensor.size m 0 in
  let rademacher = Tensor.randint 2 [n_samples; d] in
  let rademacher = Tensor.sub (Tensor.mul_scalar rademacher 2.0) 1.0 in
  let estimates = List.init n_samples (fun i ->
    let v = Tensor.select rademacher 0 i in
    let mv = Tensor.mv m v in
    Tensor.dot mv v |> Tensor.item
  ) in
  let sum = List.fold_left (+.) 0. estimates in
  sum /. float_of_int n_samples