open Torch

type sdp_solution = {
  weights: Tensor.t;
  dual_matrix: Tensor.t;
  objective: float;
}

type error_bound = {
  frobenius: float;
  spectral: float;
  multiplicative: float;
  relative: float;
}

type estimation_config = {
  epsilon: float;
  batch_size: int;
  max_iterations: int;
  memory_limit: int;
  estimation_type: [`Multiplicative | `Additive];
}