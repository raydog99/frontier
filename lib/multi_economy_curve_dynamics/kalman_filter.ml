open Torch
open Types

type kalman_state = {
  mean: Tensor.t;
  covariance: Tensor.t;
}

let predict_step model prev_state =
  let open Tensor in
  let predicted_mean = model.params.beta_0 + (model.params.beta_1 * prev_state.mean) in
  let predicted_cov = 
    matmul (matmul model.params.beta_1 prev_state.covariance) (transpose model.params.beta_1 ~dim0:0 ~dim1:1)
    + model.params.sigma_eta
  in
  { mean = predicted_mean; covariance = predicted_cov }

let update_step model predicted_state observation maturities =
  let open Tensor in
  let h = Dns_model.factor_loadings model maturities in
  let innovation = observation - (h * predicted_state.mean) in
  let s = matmul (matmul h predicted_state.covariance) (transpose h ~dim0:0 ~dim1:1) + model.params.sigma_epsilon in
  let k = matmul (matmul predicted_state.covariance (transpose h ~dim0:0 ~dim1:1)) (inverse s) in
  let updated_mean = predicted_state.mean + (matmul k innovation) in
  let updated_cov = predicted_state.covariance - (matmul (matmul k s) (transpose k ~dim0:0 ~dim1:1)) in
  { mean = updated_mean; covariance = updated_cov }

let kalman_filter model initial_state observations maturities =
  let rec filter step state acc =
    if step = Tensor.shape observations |> List.hd then
      List.rev acc
    else
      let predicted = predict_step model state in
      let observation = Tensor.select observations ~dim:0 ~index:step in
      let updated = update_step model predicted observation maturities in
      filter (step + 1) updated (updated :: acc)
  in
  let initial_kalman_state = {
    mean = initial_state;
    covariance = Tensor.eye model.state_dim
  } in
  filter 0 initial_kalman_state []

let smooth_kalman_states model filtered_states =
  let n = List.length filtered_states in
  let rec smooth step smoothed_next acc =
    if step < 0 then
      acc
    else
      let filtered = List.nth filtered_states step in
      let predicted_next = predict_step model filtered in
      let j = Tensor.(matmul filtered.covariance (transpose model.params.beta_1 ~dim0:0 ~dim1:1) 
                      / predicted_next.covariance) in
      let smoothed_mean = Tensor.(filtered.mean + (matmul j (smoothed_next.mean - predicted_next.mean))) in
      let smoothed_cov = Tensor.(filtered.covariance + 
                                 (matmul (matmul j (smoothed_next.covariance - predicted_next.covariance)) 
                                         (transpose j ~dim0:0 ~dim1:1))) in
      let smoothed = { mean = smoothed_mean; covariance = smoothed_cov } in
      smooth (step - 1) smoothed (smoothed :: acc)
  in
  let last_filtered = List.hd (List.rev filtered_states) in
  smooth (n - 2) last_filtered [last_filtered]