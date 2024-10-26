open Torch

type distribution = 
  | Binomial of { trials: Tensor.t }
  | Poisson
  | Normal of { variance: float }

type link_function =
  | Logit
  | Log 
  | Identity

type matern_params = {
  variance: float;
  range: float;
  smoothness: float;
}

type covariance_structure =
  | Matern of matern_params
  | Exponential of { variance: float; range: float }
  | Independence of { variance: float }

type model_spec = {
  distribution: distribution;
  link: link_function;
  n_obs: int;
  n_fixed: int;
  n_random: int;
  covariance: covariance_structure;
}