open Torch
open Types
open Dns_fr_model

type shock = {
  magnitude: float;
  duration: int;
  affected_maturities: int list;
}

let apply_shock reference_yields shock =
  let shocked_yields = Tensor.copy reference_yields in
  for t = 0 to shock.duration - 1 do
    let yield_slice = Tensor.select shocked_yields ~dim:0 ~index:t in
    List.iter (fun maturity ->
      let current_value = Tensor.get yield_slice [maturity] in
      Tensor.set yield_slice [maturity] (current_value +. shock.magnitude)
    ) shock.affected_maturities
  done;
  shocked_yields

let stress_test_dns_fr model initial_state reference_yields maturities shock horizon =
  let shocked_reference_yields = apply_shock reference_yields shock in
  
  (* Simulate the model under normal conditions *)
  let normal_simulation = simulate_dns_fr_model model initial_state reference_yields maturities horizon in
  
  (* Simulate the model under stressed conditions *)
  let stressed_simulation = simulate_dns_fr_model model initial_state shocked_reference_yields maturities horizon in
  
  (normal_simulation, stressed_simulation)

let calculate_stress_impact normal_simulation stressed_simulation =
  List.map2 (fun normal stressed ->
    Tensor.(stressed - normal)
  ) normal_simulation stressed_simulation

let var_stress_test model initial_state reference_yields maturities shock horizon confidence_level num_simulations =
  let shocked_reference_yields = apply_shock reference_yields shock in
  
  let simulate_with_noise yields =
    let noisy_yields = Tensor.(yields + (randn (shape yields) ~device:(device yields)) * (f 0.001)) in
    simulate_dns_fr_model model initial_state noisy_yields maturities horizon
  in
  
  let normal_simulations = List.init num_simulations (fun _ -> simulate_with_noise reference_yields) in
  let stressed_simulations = List.init num_simulations (fun _ -> simulate_with_noise shocked_reference_yields) in
  
  let calculate_var simulations =
    let stacked = Tensor.stack (List.map (fun sim -> Tensor.stack sim ~dim:0) simulations) ~dim:0 in
    let sorted_values = Tensor.sort stacked ~dim:0 ~descending:false in
    let var_index = int_of_float (float num_simulations *. (1. -. confidence_level)) in
    Tensor.select sorted_values ~dim:0 ~index:var_index
  in
  
  let normal_var = calculate_var normal_simulations in
  let stressed_var = calculate_var stressed_simulations in
  
  Tensor.(stressed_var - normal_var)

let expected_shortfall_stress_test model initial_state reference_yields maturities shock horizon confidence_level num_simulations =
  let shocked_reference_yields = apply_shock reference_yields shock in
  
  let simulate_with_noise yields =
    let noisy_yields = Tensor.(yields + (randn (shape yields) ~device:(device yields)) * (f 0.001)) in
    simulate_dns_fr_model model initial_state noisy_yields maturities horizon
  in
  
  let normal_simulations = List.init num_simulations (fun _ -> simulate_with_noise reference_yields) in
  let stressed_simulations = List.init num_simulations (fun _ -> simulate_with_noise shocked_reference_yields) in
  
  let calculate_es simulations =
    let stacked = Tensor.stack (List.map (fun sim -> Tensor.stack sim ~dim:0) simulations) ~dim:0 in
    let sorted_values = Tensor.sort stacked ~dim:0 ~descending:false in
    let es_index = int_of_float (float num_simulations *. (1. -. confidence_level)) in
    Tensor.mean (Tensor.slice sorted_values ~dim:0 ~start:(Some 0) ~end:(Some es_index) ~step:(Some 1)) ~dim:[0]
  in
  
  let normal_es = calculate_es normal_simulations in
  let stressed_es = calculate_es stressed_simulations in
  
  Tensor.(stressed_es - normal_es)