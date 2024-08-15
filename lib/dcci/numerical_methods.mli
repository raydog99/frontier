open Types
open Torch

val normal_cdf : Tensor.t -> Tensor.t
val black_scholes_call : Tensor.t -> Tensor.t -> float -> float -> Tensor.t -> Tensor.t
val fft_inplace : Complex.t array -> unit
val gauss_laguerre_quadrature : (float -> float) -> int -> float
val heston_characteristic_function_stable : model_parameters -> float -> float -> float -> Complex.t -> Complex.t
val heston_option_price_gl : model_parameters -> float -> float -> float -> option_type -> float
val sabr_volatility : float -> float -> float -> float -> float -> float -> float -> float
val calibrate_local_volatility_surface_improved : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val crank_nicolson_pde_adaptive : pde_grid -> float -> float -> float -> float -> (float -> float -> float) -> pde_grid
val fast_fourier_transform_option_pricing : (Complex.t -> Complex.t) -> float -> float -> float -> float