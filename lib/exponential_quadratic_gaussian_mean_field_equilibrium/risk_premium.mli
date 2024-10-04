open Types

val initialize_risk_premium : market_params -> risk_premium

val update_risk_premium : risk_premium -> market_params -> float -> risk_premium

val estimate_risk_premium : kalman_state -> risk_premium