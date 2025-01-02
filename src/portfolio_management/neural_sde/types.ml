open Torch

type sde_params = {
  b_v: Tensor.t -> Tensor.t -> Tensor.t;
  sigma: Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t;
  sigma_v: Tensor.t -> Tensor.t -> Tensor.t;
}

type option_data = {
  strike: float;
  maturity: float;
  market_price: float;
  weight: float;
}

type measure = [ `Risk_neutral | `Objective ]