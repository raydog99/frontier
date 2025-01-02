open Torch
open Util
open Ctmc
open Bond
open Replication
open Ross_recovery

let price_caplet ctmc t T1 T2 K state =
  let forward_rate = Bond.forward_rate ctmc t T1 T2 state in
  let discount_factor = Bond.price_at_state ctmc t T2 state in
  let payoff = Tensor.relu (Tensor.sub forward_rate (Tensor.f K)) in
  Tensor.mul discount_factor payoff

let price_floorlet ctmc t T1 T2 K state =
  let forward_rate = Bond.forward_rate ctmc t T1 T2 state in
  let discount_factor = Bond.price_at_state ctmc t T2 state in
  let payoff = Tensor.relu (Tensor.sub (Tensor.f K) forward_rate) in
  Tensor.mul discount_factor payoff

let compute_yield_curve_all_states ctmc t max_T step =
  let times = Tensor.arange (Tensor.f t) (Tensor.f max_T) (Tensor.f step) in
  let yields = Tensor.map times ~f:(fun T ->
    Tensor.stack (List.init ctmc.state_space (fun state -> Bond.yield ctmc t T state)))
  in
  (times, yields)

let limiting_yield ctmc =
  let g_minus_r = Ctmc.g_minus_r ctmc in
  let eigenvalues, _ = Tensor.symeig g_minus_r ~eigenvectors:false in
  let min_eigenvalue = Tensor.min eigenvalues in
  Tensor.neg min_eigenvalue

let price_cap ctmc t T1 T2 K state n =
  let rec sum_caplets i acc =
    if i > n then acc
    else
      let Ti = t +. (float i) *. (T2 -. T1) /. (float n) in
      let caplet_price = price_caplet ctmc t Ti (Ti +. (T2 -. T1) /. (float n)) K state in
      sum_caplets (i + 1) (Tensor.add acc caplet_price)
  in
  sum_caplets 1 (Tensor.zeros [])

let price_floor ctmc t T1 T2 K state n =
  let rec sum_floorlets i acc =
    if i > n then acc
    else
      let Ti = t +. (float i) *. (T2 -. T1) /. (float n) in
      let floorlet_price = price_floorlet ctmc t Ti (Ti +. (T2 -. T1) /. (float n)) K state in
      sum_floorlets (i + 1) (Tensor.add acc floorlet_price)
  in
  sum_floorlets 1 (Tensor.zeros [])