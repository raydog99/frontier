open Torch
open Types

let create_dns_model lambda state_dim obs_dim =
  let params = {
    lambda = lambda;
    beta_0 = Tensor.zeros [state_dim];
    beta_1 = Tensor.zeros [state_dim; state_dim];
    sigma_epsilon = Tensor.eye obs_dim;
    sigma_eta = Tensor.eye state_dim;
  } in
  { params; state_dim; obs_dim }

let factor_loadings model maturities =
  let open Tensor in
  let lambda = model.params.lambda in
  let exp_term = Tensor.( -lambda * maturities) |> exp in
  let level = ones_like maturities in
  let slope = Tensor.((f 1. - exp_term) / (lambda * maturities)) in
  let curvature = Tensor.((f 1. - exp_term) / (lambda * maturities) - exp_term) in
  stack [level; slope; curvature] ~dim:0

let measurement_equation model state maturities =
  let loadings = factor_loadings model maturities in
  Tensor.(loadings * state)

let transition_equation model prev_state =
  let open Tensor in
  model.params.beta_0 + (model.params.beta_1 * prev_state)

let simulate_dns_model model initial_state maturities num_steps =
  let rec simulate step state acc =
    if step = num_steps then
      List.rev acc
    else
      let new_state = transition_equation model state in
      let observation = measurement_equation model new_state maturities in
      simulate (step + 1) new_state (observation :: acc)
  in
  simulate 0 initial_state []