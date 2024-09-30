open Torch
open Util
open Ctmc

let price ctmc t T =
  check_valid_time t "t";
  check_valid_time T "T";
  if t >= T then failwith "t must be less than T";
  let dt = Tensor.sub (Tensor.f T) (Tensor.f t) in
  let g_minus_r = g_minus_r ctmc in
  matrix_exp g_minus_r dt

let price_at_state ctmc t T state =
  check_valid_state ctmc state;
  let prices = price ctmc t T in
  Tensor.select prices 0 state