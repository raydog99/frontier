open Torch

type asset = {
  symbol: string;
  prices: Tensor.t;
}

type portfolio = {
  assets: asset list;
  weights: Tensor.t;
}