open Torch

type yield_curve = Tensor.t

type state_vector = Tensor.t

type observation_vector = Tensor.t

type parameters = {
  lambda: float;
  beta_0: Tensor.t;
  beta_1: Tensor.t;
  sigma_epsilon: Tensor.t;
  sigma_eta: Tensor.t;
}

type model = {
  params: parameters;
  state_dim: int;
  obs_dim: int;
}