open Torch

let create_bernoulli_matrix n p =
  Utils.assert_positive_float "n" (float_of_int n);
  Utils.assert_in_range "p" p 0. 1.;
  let bernoulli = Tensor.rand [n; n] ~dtype:Kind.Float in
  let mask = Tensor.lt bernoulli (Tensor.float_scalar p) in
  Tensor.where mask (Tensor.ones [n; n] ~dtype:Kind.Float) (Tensor.zeros [n; n] ~dtype:Kind.Float)

let bernoulli_bulk_overlap q p =
  Utils.assert_in_range "q" q 0. 1.;
  Utils.assert_in_range "p" p 0. 1.;
  let t = p *. (1. -. p) in
  Overlap_analysis.limiting_rescaled_mean_squared_overlap q t

let bernoulli_spike_overlap n q p =
  Utils.assert_positive_float "n" (float_of_int n);
  Utils.assert_in_range "q" q 0. 1.;
  Utils.assert_in_range "p" p 0. 1.;
  let lambda_n = p *. (sqrt (float_of_int n)) in
  let mu_n = q *. lambda_n in
  let t = p *. (1. -. p) in
  Tensor.div
    (Tensor.mul_scalar (Tensor.float_scalar q) (Tensor.float_scalar (1. -. q)))
    (Tensor.add
      (Tensor.pow (Tensor.float_scalar (1. -. q)) (Scalar.float 2.0))
      (Tensor.div
        (Tensor.float_scalar t)
        (Tensor.pow (Tensor.float_scalar p) (Scalar.float 2.0))))