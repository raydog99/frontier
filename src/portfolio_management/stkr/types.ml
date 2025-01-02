open Torch

type kernel = Tensor.t -> Tensor.t -> Tensor.t
type transform_fn = float -> float

type stkr_params = {
  lambda: float;
  epsilon: float;
  max_iter: int;
  learning_rate: float;
  batch_size: int option;
}

type polynomial_params = {
  degree: int;
  coefficients: float array;
}

type sparse_matrix = {
  values: Tensor.t;
  indices: (int * int) array;
  shape: int * int;
}