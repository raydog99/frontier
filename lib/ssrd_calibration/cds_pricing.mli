open Types

val payment_dates : float -> float -> float list
val price_cds : params -> float -> float -> float -> approximation_order -> float
val calibrate_ir_model : (float * float) list -> float * float * float
val match_zcb_price : params -> float -> float -> float
val calibrate_to_market_data : params -> market_data -> approximation_order -> optimization_result