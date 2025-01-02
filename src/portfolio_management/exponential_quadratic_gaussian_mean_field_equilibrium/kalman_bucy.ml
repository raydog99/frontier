open Torch
open Types

let initialize_kalman_state params =
  let estimate = Tensor.of_float1 [|params.m|] in
  let error_covariance = Tensor.of_float1 [|params.v|] in
  { estimate; error_covariance }

let predict kalman_state params dt =
  let { estimate; error_covariance } = kalman_state in
  let predicted_estimate = Tensor.(add (mul estimate (add (float 1.) (mul params.alpha (float dt))))
                                       (mul params.beta (float dt))) in
  let predicted_error_covariance = Tensor.(add (mul error_covariance (add (float 1.) (mul (mul (float 2.) params.alpha) (float dt))))
                                               (mul (add (mul params.zeta params.zeta) (mul params.eta params.eta)) (float dt))) in
  { estimate = predicted_estimate; error_covariance = predicted_error_covariance }

let update kalman_state observation params dt =
  let { estimate; error_covariance } = kalman_state in
  let innovation = Tensor.(sub observation (mul estimate (float dt))) in
  let innovation_covariance = Tensor.(add error_covariance (mul (float params.sigma0) (float params.sigma0))) in
  let kalman_gain = Tensor.(div error_covariance innovation_covariance) in
  let updated_estimate = Tensor.(add estimate (mul kalman_gain innovation)) in
  let updated_error_covariance = Tensor.(sub error_covariance (mul kalman_gain error_covariance)) in
  { estimate = updated_estimate; error_covariance = updated_error_covariance }

let estimate_risk_premium kalman_state =
  kalman_state.estimate