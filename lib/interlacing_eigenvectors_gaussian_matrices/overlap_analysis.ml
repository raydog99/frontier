open Torch

let limiting_rescaled_mean_squared_overlap q t mu lambda =
  Utils.assert_in_range "q" q 0. 1.;
  Utils.assert_positive_float "t" t;
  let numerator = Tensor.mul_scalar (Tensor.float_scalar ((1. -. q) *. t)) lambda in
  let denominator_term1 = Tensor.pow (Tensor.mul_scalar (Tensor.sub lambda mu) (Tensor.float_scalar (1. -. q))) (Scalar.float 2.0) in
  let denominator_term2 = Tensor.mul_scalar (Tensor.mul lambda mu) (Tensor.float_scalar t) in
  let denominator = Tensor.add denominator_term1 denominator_term2 in
  Utils.safe_div numerator denominator

let find_optimal_overlap q t mu =
  Utils.assert_in_range "q" q 0. 1.;
  Utils.assert_positive_float "t" t;
  let lambda = Tensor.arange (-.2. *. (sqrt t)) (2. *. (sqrt t)) ~step:0.01 in
  let overlaps = limiting_rescaled_mean_squared_overlap q t mu lambda in
  let max_overlap, max_index = Tensor.max overlaps ~dim:0 ~keepdim:false in
  Tensor.select lambda 0 (Tensor.item max_index)

let check_interlacing q t mu lambda_star =
  Utils.assert_in_range "q" q 0. 1.;
  Utils.assert_positive_float "t" t;
  let lambda_q_x = Tensor.mul_scalar mu (Tensor.float_scalar (1. /. (sqrt q))) in
  let lambda_q_x_plus_1_minus_q = Tensor.add lambda_q_x (Tensor.float_scalar ((1. -. q) *. 2. *. (sqrt t))) in
  Tensor.ge lambda_star lambda_q_x && Tensor.le lambda_star lambda_q_x_plus_1_minus_q

let compute_overlap_distribution n q t lambda =
  Utils.assert_positive_float "n" (float_of_int n);
  Utils.assert_in_range "q" q 0. 1.;
  Utils.assert_positive_float "t" t;
  let mu = Tensor.arange (-.2. *. (sqrt (q *. t))) (2. *. (sqrt (q *. t))) ~step:0.01 in
  let overlaps = limiting_rescaled_mean_squared_overlap q t mu lambda in
  let normalization = Tensor.sum overlaps in
  Utils.safe_div overlaps normalization

let compute_bulk_edge q t =
  Utils.assert_in_range "q" q 0. 1.;
  Utils.assert_positive_float "t" t;
  2. *. sqrt (q *. t)