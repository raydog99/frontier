open Torch

type asset = {
  id: int;
  returns: Tensor.t;
}

type portfolio = {
  assets: asset array;
  weights: Tensor.t;
  expected_returns: Tensor.t;
}

type covariance_matrix = Tensor.t

type correlation_matrix = Tensor.t

type community = int array

type decomposition_result = {
  communities: community array;
  subproblems: portfolio array;
}

type eigen_decomposition = {
  eigenvalues: Tensor.t;
  eigenvectors: Tensor.t;
}

type optimization_params = {
  risk_aversion: float;
  max_iterations: int;
  tolerance: float;
  cardinality_constraint: int option;
}

type optimization_result = {
  weights: Tensor.t;
  objective_value: float;
  iterations: int;
}

exception OptimizationError of string