open Torch

let mean data =
  let tensor = Tensor.of_float1 data in
  Tensor.mean tensor |> Tensor.to_float0_exn

let variance data =
  let tensor = Tensor.of_float1 data in
  Tensor.var tensor ~unbiased:true |> Tensor.to_float0_exn

let standard_deviation data =
  let tensor = Tensor.of_float1 data in
  Tensor.std tensor ~unbiased:true |> Tensor.to_float0_exn

let covariance_matrix returns =
  let tensor = Tensor.of_float2 returns in
  let centered = Tensor.sub tensor (Tensor.mean tensor ~dim:[1] ~keepdim:true) in
  let n = float_of_int (Tensor.shape tensor |> List.hd) in
  Tensor.matmul (Tensor.transpose centered ~dim0:0 ~dim1:1) centered |> Tensor.div_scalar (n -. 1.)

let correlation_matrix returns =
  let cov = covariance_matrix returns in
  let std_devs = Tensor.sqrt (Tensor.diag cov) in
  let outer_std = Tensor.matmul (Tensor.unsqueeze std_devs ~dim:1) (Tensor.unsqueeze std_devs ~dim:0) in
  Tensor.div cov outer_std