open Torch
open Types

let normal_cdf x =
  let half = Scalar.float 0.5 in
  let one = Scalar.float 1.0 in
  Tensor.(half * (one + erf (x / sqrt (Scalar.float 2.0))))

let black_scholes_call s k r t sigma =
  let d1 = Tensor.((log (s / k) + (r + sigma * sigma / Scalar.float 2.0) * t) / (sigma * sqrt t)) in
  let d2 = Tensor.(d1 - sigma * sqrt t) in
  Tensor.(s * normal_cdf d1 - k * exp (Scalar.neg r * t) * normal_cdf d2)

let generate_correlation_matrix beta num_futures =
  let matrix = Tensor.zeros [num_futures; num_futures] in
  for i = 0 to num_futures - 1 do
    for j = 0 to num_futures - 1 do
      let correlation = exp (-. beta *. float (abs (i - j))) in
      Tensor.set matrix [|i; j|] (Scalar.float correlation)
    done
  done;
  matrix

let compute_index_value weights futures_prices =
  Tensor.(sum (weights * futures_prices))

let create_time_grid start_date end_date num_steps =
  let dt = (end_date -. start_date) /. float_of_int num_steps in
  Array.init (num_steps + 1) (fun i -> start_date +. float_of_int i *. dt)

let interpolate_vol surface t k =
  Tensor.zeros []

let mollifier x epsilon =
  if Tensor.(abs x < Scalar.float epsilon) then
    Tensor.(exp (Scalar.float (-1.0) / (Scalar.float 1.0 - pow (x / Scalar.float epsilon) (Scalar.float 2.0))))
  else
    Tensor.zeros_like x

let create_log_space start stop num =
  let log_start = log start in
  let log_stop = log stop in
  let step = (log_stop -. log_start) /. (float_of_int (num - 1)) in
  Array.init num (fun i -> exp (log_start +. float_of_int i *. step))

let create_pde_grid x_min x_max t_max nx nt =
  let x = Tensor.linspace x_min x_max nx in
  let t = Tensor.linspace 0. t_max nt in
  let v = Tensor.zeros [nt; nx] in
  { x; t; v }

let tridiag_solver a b c d =
  let n = Tensor.shape d |> List.hd in
  let x = Tensor.zeros_like d in
  let c_prime = Tensor.zeros_like c in
  
  Tensor.set c_prime [|0|] (Tensor.get c [|0|]);
  Tensor.set x [|0|] (Tensor.get d [|0|] / Tensor.get b [|0|]);
  
  for i = 1 to n - 1 do
    let m = Tensor.get a [|i|] / Tensor.get c_prime [|i-1|] in
    Tensor.set c_prime [|i|] (Tensor.get c [|i|] - m * Tensor.get b [|i-1|]);
    Tensor.set x [|i|] ((Tensor.get d [|i|] - m * Tensor.get x [|i-1|]) / Tensor.get c_prime [|i|])
  done;
  
  for i = n - 2 downto 0 do
    Tensor.set x [|i|] (Tensor.get x [|i|] - Tensor.get c [|i|] / Tensor.get c_prime [|i|] * Tensor.get x [|i+1|])
  done;
  
  x

let compute_index_weights index_params futures_contracts t =
  let { roll_schedule; calculation_method } = index_params in
  match calculation_method with
  | StandardRolling ->
      let i = Array.fold_left (fun acc date -> if t >= date then acc + 1 else acc) 0 roll_schedule.roll_dates in
      if i >= Array.length roll_schedule.front_contract_weights then
        [| 0.; 1. |]
      else
        [| roll_schedule.front_contract_weights.(i); roll_schedule.back_contract_weights.(i) |]
  | EnhancedRolling ->
      [| 0.5; 0.5 |]

let calculate_index_value index_params futures_contracts t =
  let weights = compute_index_weights index_params futures_contracts t in
  let front_price = futures_contracts.(0).price in
  let back_price = futures_contracts.(1).price in
  weights.(0) *. front_price +. weights.(1) *. back_price