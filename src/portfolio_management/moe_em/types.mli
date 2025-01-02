open Torch
type expert_type = Linear | Logistic

type config = {
  n_experts: int;
  input_dim: int;
  expert_type: expert_type;
  symmetric: bool;
}

type model = {
  experts: Tensor.t;
  gates: Tensor.t;
  config: config;
}

type mirror_map = {
  linear_map: Tensor.t -> float;
  logistic_map: Tensor.t -> float;
}

type optimization_config = {
  max_iterations: int;
  tolerance: float;
  relative_smoothness: float;
}

type optimization_state = {
  iteration: int;
  loss: float;
  kl_divergence: float;
  converged: bool;
}

type fisher_info = {
  complete_data: Tensor.t;
  conditional: Tensor.t;
}

type convergence_metrics = {
  mim_max_eigenvalue: float;
  relative_convexity: float;
  snr_estimate: float;
  convergence_rate: float option;
}

type theorem_conditions = {
  locally_convex: bool;
  relatively_strongly_convex: bool;
  sufficient_snr: bool;
}