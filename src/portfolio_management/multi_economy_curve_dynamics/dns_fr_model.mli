open Types
open Functional_regression
open Kalman_filter
open Torch

type dns_fr_model = {
  dns: model;
  kpca: kpca_result;
  beta: Tensor.t;
  n_components: int;
  sigma: float;
}

val create_dns_fr_model : float -> int -> int -> int -> float -> dns_fr_model

val fit_dns_fr_model : dns_fr_model -> Tensor.t -> Tensor.t -> Tensor.t -> dns_fr_model

val measurement_equation_fr : dns_fr_model -> state_vector -> Tensor.t -> Tensor.t -> observation_vector

val simulate_dns_fr_model : dns_fr_model -> state_vector -> Tensor.t -> Tensor.t -> int -> observation_vector list

val kalman_filter_fr : dns_fr_model -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> kalman_state list

val forecast_dns_fr_model : dns_fr_model -> state_vector -> Tensor.t -> Tensor.t -> int -> observation_vector list