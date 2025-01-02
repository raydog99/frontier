open Torch
open Types

exception InvalidData of string

val prepare_data : market_index -> Tensor.t
val calculate_dcca_matrix : Tensor.t array -> int -> float array array
val calculate_dccc : Tensor.t array -> Tensor.t array -> int -> int -> float -> dccc_result
val rolling_window_analysis : market_index array -> analysis_config -> analysis_result