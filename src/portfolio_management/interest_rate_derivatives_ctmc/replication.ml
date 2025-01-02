open Torch
open Util
open Ctmc
open Arrow_debreu
open Bond

let delta ctmc t T1 T k state =
  check_valid_state ctmc state;
  check_valid_state ctmc (state + 1);
  let ad_price_diff = Tensor.sub 
    (price_at_state ctmc t T (state + 1)) 
    (price_at_state ctmc t T state) in
  let bond_price_diff = Tensor.sub 
    (price_at_state ctmc t T1 (state + 1)) 
    (price_at_state ctmc t T1 state) in
  Tensor.div ad_price_diff bond_price_diff

let replicate_arrow_debreu ctmc t T1 T k initial_state =
  check_valid_time t "t";
  check_valid_time T1 "T1";
  check_valid_time T "T";
  check_valid_state ctmc initial_state;
  if t >= T1 || T1 >= T then failwith "Must have t < T1 < T";
  
  let rec simulate current_t current_state value =
    if current_t >= T then value
    else
      let dt = min (T -. current_t) (1. /. (jump_intensity ctmc current_state |> Tensor.item)) in
      let new_t = current_t +. dt in
      let delta = delta ctmc (Tensor.f current_t) (Tensor.f T1) (Tensor.f T) k current_state in
      let bond_price = price_at_state ctmc (Tensor.f current_t) (Tensor.f T1) current_state in
      let new_bond_price = price_at_state ctmc (Tensor.f new_t) (Tensor.f T1) current_state in
      let rate = ctmc.rates |> Tensor.select 0 current_state |> Tensor.item in
      let new_value = value *. (1. +. rate *. dt) +. (Tensor.item delta) *. (Tensor.item new_bond_price -. Tensor.item bond_price) in
      let jump_probs = jump_probabilities ctmc current_state in
      let new_state = 
        if Random.float 1. < (jump_intensity ctmc current_state |> Tensor.item) *. dt
        then Tensor.multinomial jump_probs ~num_samples:1 |> Tensor.int_repr |> Tensor.item
        else current_state 
      in
      simulate new_t new_state new_value
  in
  let initial_value = price_at_state ctmc (Tensor.f t) (Tensor.f T) k |> Tensor.item in
  simulate t initial_state initial_value