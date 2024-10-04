open Torch
open Types

val solve_eqg : market_params -> ode_solution -> float -> eqg_solution

val calculate_value_function : eqg_solution -> Tensor.t -> Tensor.t -> Tensor.t