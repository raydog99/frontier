open Torch
open Util
open Ctmc
open Arrow_debreu

let price ctmc t T =
  let prices = price ctmc t T in
  Tensor.sum prices ~dim:[1] ~keepdim:true

let price_at_state ctmc t T state =
  check_valid_state ctmc state;
  let prices = price ctmc t T in
  Tensor.select prices 0 state

let yield ctmc t T state =
  let p = price_at_state ctmc t T state in
  Tensor.div (Tensor.neg (Tensor.log p)) (Tensor.sub (Tensor.f T) (Tensor.f t))

let forward_rate ctmc t T1 T2 state =
  check_valid_time t "t";
  check_valid_time T1 "T1";
  check_valid_time T2 "T2";
  check_valid_state ctmc state;
  if t >= T1 || T1 >= T2 then failwith "Must have t < T1 < T2";
  let p1 = price_at_state ctmc t T1 state in
  let p2 = price_at_state ctmc t T2 state in
  let f = Tensor.div (Tensor.log (Tensor.div p1 p2)) (Tensor.sub (Tensor.f T2) (Tensor.f T1)) in
  Tensor.neg f