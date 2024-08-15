open Torch
open Types

val normal_cdf : Tensor.t -> Tensor.t
val black_scholes_call : Tensor.t -> Tensor.t -> float -> float -> Tensor.t -> Tensor.t
val generate_correlation_matrix : float -> int -> Tensor.t
val compute_index_value : Tensor.t -> Tensor.t -> Tensor.t
val create_time_grid : float -> float -> int -> float array
val interpolate_vol : Tensor.t -> float -> float -> Tensor.t
val mollifier : Tensor.t -> float -> Tensor.t
val create_log_space : float -> float -> int -> float array
val create_pde_grid : float -> float -> float -> int -> int -> pde_grid
val tridiag_solver : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val compute_index_weights : index_calculation_parameters -> futures_contract array -> float -> float array
val calculate_index_value : index_calculation_parameters -> futures_contract array -> float -> float