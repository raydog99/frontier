open Types

val detect_regime : float -> float -> float -> regime
val detect_regimes : dccc_result array -> float -> float -> regime_detection_result array
val identify_influential_markets : float array array -> market_index array -> (string * float) list
val analyze_market_efficiency : market_index array -> market_efficiency array
val detect_volatility_clusters : float array -> int -> float -> volatility_cluster array
val analyze_market_risk : market_index array -> market_index -> (string * float) array