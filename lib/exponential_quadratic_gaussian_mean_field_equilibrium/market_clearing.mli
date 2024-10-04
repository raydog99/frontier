open Torch
open Types

val calculate_average_strategy : Tensor.t list -> Tensor.t

val calculate_clearing_error : Tensor.t -> float

val is_market_cleared : Tensor.t -> float -> bool

val update_market_clearing_state : Tensor.t list -> market_clearing_state

val adjust_risk_premium : risk_premium -> market_clearing_state -> market_params -> risk_premium