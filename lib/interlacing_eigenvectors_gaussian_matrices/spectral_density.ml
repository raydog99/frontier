open Torch

let wigner_semicircle_density t x =
  Utils.assert_positive_float "t" t;
  let r = 2. *. (sqrt t) in
  if Tensor.item x >= -.r && Tensor.item x <= r then
    let y = Tensor.pow x (Scalar.float 2.0) in
    Tensor.div (Tensor.sqrt (Tensor.sub (Tensor.float_scalar (r ** 2.)) y)) (Tensor.float_scalar (2. *. Float.pi *. t))
  else
    Tensor.zeros_like x

let wigner_semicircle_stieltjes t z =
  Utils.assert_positive_float "t" t;
  let z_squared = Tensor.pow z (Scalar.float 2.0) in
  Tensor.div
    (Tensor.sub z (Tensor.sqrt (Tensor.sub z_squared (Tensor.float_scalar (4. *. t)))))
    (Tensor.mul_scalar (Tensor.float_scalar (2. *. t)) z)

let marcenko_pastur_density c t x =
  Utils.assert_positive_float "c" c;
  Utils.assert_positive_float "t" t;
  let lambda_plus = t *. (1. +. sqrt c) ** 2. in
  let lambda_minus = t *. (1. -. sqrt c) ** 2. in
  let x_val = Tensor.item x in
  if x_val < lambda_minus || x_val > lambda_plus then
    Tensor.zeros_like x
  else
    let numerator = sqrt ((lambda_plus -. x_val) *. (x_val -. lambda_minus)) in
    let denominator = 2. *. Float.pi *. t *. x_val in
    Tensor.float_scalar (numerator /. denominator)

let compute_dos eigenvalues num_points min_e max_e =
  let hist = Tensor.histc eigenvalues ~bins:num_points ~min:min_e ~max:max_e in
  let bin_width = (max_e -. min_e) /. float_of_int num_points in
  Utils.safe_div hist (Tensor.float_scalar (bin_width *. float_of_int (Tensor.size eigenvalues ~dim:0)))