open Torch

type distribution = {
  mean: Tensor.t;
  covariance: Tensor.t;
}

type markov_chain = {
  kernel: Tensor.t -> Tensor.t;
  stationary_dist: Tensor.t -> float;
}