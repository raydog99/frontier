open Types
open Torch
open Lwt

module MacroModel : sig
  val simulate_parallel : model_parameters -> market_data -> int -> int -> Tensor.t Lwt.t
  val price_vanilla_option : model_parameters -> market_data -> float -> float -> option_type -> (Tensor.t * Tensor.t) Lwt.t
  val price_path_dependent_option : model_parameters -> market_data -> path_dependent_option -> int -> int -> float Lwt.t
  val compute_var : model_parameters -> market_data -> float -> float -> Tensor.t Lwt.t
  val compute_expected_shortfall : model_parameters -> market_data -> float -> float -> Tensor.t Lwt.t
  val calibrate_stochastic_volatility : market_data -> float * float * float * float
  val price_with_characteristic_function : model_parameters -> market_data -> option_type -> float -> float -> float
end