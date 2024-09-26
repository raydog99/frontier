open Torch

let spike_trajectory lambda t =
  Utils.assert_positive_float "lambda" (Tensor.item lambda);
  Utils.assert_positive_float "t" (Tensor.item t);
  Tensor.add lambda (Tensor.div t lambda)

let spike_spike_overlap mu lambda t q =
  Utils.assert_positive_float "mu" (Tensor.item mu);
  Utils.assert_positive_float "lambda" (Tensor.item lambda);
  Utils.assert_positive_float "t" (Tensor.item t);
  Utils.assert_in_range "q" q 0. 1.;
  let numerator = Tensor.mul mu (Tensor.sub (Tensor.pow lambda (Scalar.float 2.0)) t) in
  let denominator_term1 = Tensor.pow (Tensor.sub (Tensor.mul mu lambda) (Tensor.mul_scalar t (Tensor.float_scalar q))) (Scalar.float 2.0) in
  let denominator_term2 = Tensor.mul (Tensor.pow mu (Scalar.float 2.0)) (Tensor.mul_scalar t (Tensor.float_scalar q)) in
  let denominator = Tensor.sub denominator_term1 denominator_term2 in
  Utils.safe_div numerator denominator

let spike_bulk_overlap mu lambda t q =
  Utils.assert_positive_float "mu" (Tensor.item mu);
  Utils.assert_positive_float "lambda" (Tensor.item lambda);
  Utils.assert_positive_float "t" (Tensor.item t);
  Utils.assert_in_range "q" q 0. 1.;
  let numerator = Tensor.mul_scalar (Tensor.sub (Tensor.pow lambda (Scalar.float 2.0)) (Tensor.mul_scalar t (Tensor.float_scalar q))) t in
  let denominator = Tensor.pow (Tensor.add (Tensor.mul_scalar lambda (Tensor.float_scalar 2.)) (Tensor.sub (Tensor.mul_scalar t (Tensor.float_scalar q)) mu)) (Scalar.float 2.0) in
  Utils.safe_div numerator denominator

let compute_critical_time lambda =
  Utils.assert_positive_float "lambda" lambda;
  lambda ** 2.

let compute_spike_phase_transition lambda t =
  Utils.assert_positive_float "lambda" lambda;
  Utils.assert_positive_float "t" t;
  let critical_lambda = sqrt t in
  if lambda > critical_lambda then
    lambda +. (t /. lambda)
  else
    2. *. sqrt t