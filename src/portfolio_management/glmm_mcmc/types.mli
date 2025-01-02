open Torch

type data = {
  y: Tensor.t;
  x: Tensor.t;
  z: Tensor.t;
  trials: int option;  (* For binomial models *)
}

type model_params = {
  beta: Tensor.t;
  u: Tensor.t;
  lambda: Tensor.t;
}

type prior_params = {
  mu_0: Tensor.t;
  q: Tensor.t;
  a: Tensor.t;
  b: Tensor.t;
}

type mcmc_state = {
  params: model_params;
  log_prob: float;
  accepted: int;
  total: int;
  epsilon: float;
}