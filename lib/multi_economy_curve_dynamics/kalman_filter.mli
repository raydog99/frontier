open Types
open Torch

type kalman_state = {
  mean: Tensor.t;
  covariance: Tensor.t;
}

val predict_step : model -> kalman_state -> kalman_state

val update_step : model -> kalman_state -> Tensor.t -> Tensor.t -> kalman_state

val kalman_filter : model -> Tensor.t -> Tensor.t -> Tensor.t -> kalman_state list

val smooth_kalman_states : model -> kalman_state list -> kalman_state list