open Torch
open Base

type garch_variant = GARCH | GJR_GARCH | TGARCH

type garch_params = {
  alpha_0: float;
  alpha: float;
  beta: float;
  gamma: float option;
  variant: garch_variant;
}

type time_series = float array

type model_type = GINN | GINN_0

val garch_predict : garch_params -> time_series -> int -> float array

val create_lstm_model : int -> int -> int -> (Tensor.t -> Tensor.t) Staged.t

val create_ginn_model : int -> int -> int -> Tensor.t -> model_type -> (Tensor.t -> Tensor.t -> Tensor.t) Staged.t

val generate_random_series : int -> float array

val calculate_returns : float array -> float array

val normalize_data : Tensor.t -> Tensor.t * Tensor.t * Tensor.t

val prepare_data : time_series -> int -> (Tensor.t * Tensor.t) * (Tensor.t * Tensor.t) * (Tensor.t * Tensor.t)

val ginn_loss : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

val ginn_0_loss : Tensor.t -> Tensor.t -> Tensor.t

val train_ginn_model : (Tensor.t -> Tensor.t -> Tensor.t) Staged.t -> garch_params -> time_series -> int -> int -> float -> Tensor.t -> model_type -> unit

val evaluate_model : (Tensor.t -> Tensor.t -> Tensor.t) Staged.t -> garch_params -> time_series -> int -> model_type -> float * float * float