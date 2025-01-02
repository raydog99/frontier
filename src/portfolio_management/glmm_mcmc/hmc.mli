open Types

type hmc_state = {
  position: model_params;
  momentum: model_params;
  log_prob: float;
}

val step : data -> hmc_state -> float -> int -> model_params
val rmhmc_step : data -> hmc_state -> float -> int -> model_params
val nuts_step : data -> hmc_state -> float -> int -> model_params