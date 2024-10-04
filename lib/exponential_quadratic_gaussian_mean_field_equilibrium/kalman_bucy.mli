open Torch
open Types

val initialize_kalman_state : market_params -> kalman_state

val predict : kalman_state -> market_params -> float -> kalman_state

val update : kalman_state -> Tensor.t -> market_params -> float -> kalman_state

val estimate_risk_premium : kalman_state -> risk_premium