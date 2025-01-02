open Torch
open Types

let initialize_risk_premium params =
  Tensor.randn [1] ~mean:(Scalar.float params.m) ~std:(Scalar.float (sqrt params.v))

let update_risk_premium risk_premium params dt =
  let dW = Tensor.randn [1] ~mean:(Scalar.float 0.) ~std:(Scalar.float (sqrt dt)) in
  let dB = Tensor.randn [1] ~mean:(Scalar.float 0.) ~std:(Scalar.float (sqrt dt)) in
  
  Tensor.(add risk_premium
    (add (mul (add (mul params.alpha risk_premium) params.beta) (float dt))
         (add (mul params.zeta dW) (mul params.eta dB))))

let estimate_risk_premium kalman_state =
  kalman_state.estimate