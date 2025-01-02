open Types
open Torch

val compute_background_charge : genus -> float
val compute_area : conformal_factor -> area
val compute_physical_time : background_time -> conformal_factor -> physical_time
val evolve_conformal_factor : conformal_factor -> genus -> float -> conformal_factor
val evolve_area : area -> genus -> float -> area
val evolve_genus : genus -> area -> genus
val compute_equilibrium_distribution : genus -> int -> Tensor.t
val compute_expectation_value : (Tensor.t -> Tensor.t) -> Tensor.t -> float
val compute_average_genus : Tensor.t -> float
val correlation_function_zero_mode : float list -> float list -> genus -> float
val correlation_function_nonzero_modes : float list -> float list -> float
val full_correlation_function : float list -> float list -> genus -> float