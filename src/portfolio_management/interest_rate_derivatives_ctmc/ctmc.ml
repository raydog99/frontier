open Torch
open Util

type t = {
  generator: Tensor.t;
  rates: Tensor.t;
  state_space: int;
}

let create generator rates =
  check_square_matrix generator;
  check_square_matrix rates;
  check_nonnegative_tensor rates "Interest rates";
  let state_space = Tensor.size generator 0 in
  if Tensor.size rates 0 != state_space then
    failwith "Generator and rates matrices must have the same dimensions";
  { generator; rates; state_space }

let matrix_exp m t =
  check_positive_scalar t "Time";
  Tensor.matrix_exp (Tensor.mul_scalar m (Tensor.f t))

let g_minus_r ctmc =
  Tensor.sub ctmc.generator ctmc.rates

let jump_intensity ctmc state =
  check_valid_state ctmc state;
  Tensor.neg (Tensor.select (Tensor.diag ctmc.generator) 0 state)

let jump_probabilities ctmc state =
  check_valid_state ctmc state;
  let row = Tensor.select ctmc.generator 0 state in
  let probs = Tensor.div row (jump_intensity ctmc state) in
  Tensor.set probs state 0.;
  probs