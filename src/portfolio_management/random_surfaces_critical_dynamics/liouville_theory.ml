open Torch
open Types
open Constants
open Utils
open Logging

let compute_background_charge genus =
  q *. float_of_int (1 - genus)

let compute_area conformal_factor =
  exp (alpha *. conformal_factor)

let compute_physical_time background_time conformal_factor =
  background_time *. exp (alpha *. conformal_factor)

let evolve_conformal_factor conformal_factor genus dt =
  let drift = 0.5 *. (compute_background_charge genus -. cosmological_constant *. alpha *. exp (alpha *. conformal_factor)) in
  let volatility = 1.0 in
  geometric_brownian_motion conformal_factor drift volatility dt

let evolve_area current_area genus dt =
  let a = cosmological_constant *. alpha *. alpha /. 2.0 in
  let b = q /. (cosmological_constant *. alpha) *. float_of_int (genus - 1) in
  let cir_process a b alpha current_value dt =
    let drift = a *. (b -. current_value) in
    let volatility = alpha *. sqrt current_value in
    let noise = Tensor.randn [1] in
    max 0.0 (current_value +. drift *. dt +. volatility *. Tensor.get noise [0] *. sqrt dt)
  in
  cir_process a b cir_alpha current_area dt

let evolve_genus genus area =
  let probabilities = [|0.1; 0.8; 0.1|] in
  let random_value = Random.float 1.0 in
  let new_genus = 
    if random_value < probabilities.(0) then max 0 (genus - 1)
    else if random_value < probabilities.(0) +. probabilities.(1) then genus
    else genus + 1
  in
  min new_genus (int_of_float (log area /. log cosmological_constant))

let compute_equilibrium_distribution genus num_samples =
  let area_samples = Tensor.linspace 0.0 10.0 num_samples in
  let nu = (q /. alpha) *. float_of_int (genus - 1) in
  Tensor.pow area_samples nu *. Tensor.exp (Tensor.neg (Tensor.of_float cosmological_constant *. area_samples))

let compute_expectation_value f distribution =
  let weighted_values = Tensor.mul (f distribution) distribution in
  Tensor.sum weighted_values /. Tensor.sum distribution

let compute_average_genus distribution =
  let genus_function areas =
    Tensor.floor (Tensor.div (Tensor.log areas) (Tensor.log (Tensor.of_float cosmological_constant)))
  in
  compute_expectation_value genus_function distribution

let correlation_function_zero_mode times charges genus =
  let n = List.length times in
  let result = ref 1.0 in
  let t_mean = List.fold_left (+.) 0.0 times /. float_of_int n in
  for i = 0 to n - 1 do
    for j = i + 1 to n - 1 do
      let t_i = List.nth times i in
      let t_j = List.nth times j in
      let charge_i = List.nth charges i in
      let charge_j = List.nth charges j in
      result := !result *. (abs_float (t_i -. t_j) ** (charge_i *. charge_j))
    done;
    let t_i = List.nth times i in
    let charge_i = List.nth charges i in
    result := !result *. exp (-. charge_i *. q *. (1.0 -. float_of_int genus) *. (t_mean -. t_i))
  done;
  !result

let correlation_function_nonzero_modes times charges =
  let n = List.length times in
  let result = ref 1.0 in
  for i = 0 to n - 1 do
    for j = i + 1 to n - 1 do
      let t_i = List.nth times i in
      let t_j = List.nth times j in
      let charge_i = List.nth charges i in
      let charge_j = List.nth charges j in
      result := !result *. (abs_float (t_i -. t_j) ** (-. charge_i *. charge_j))
    done
  done;
  !result

let full_correlation_function times charges genus =
  (correlation_function_zero_mode times charges genus) *. (correlation_function_nonzero_modes times charges)