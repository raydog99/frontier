open Torch

type task_data = {
  x: Tensor.t;  (** n_l x p data matrix *)
  y: Tensor.t;  (** n_l dimensional vector *)
}

type model_params = {
  sigma: Tensor.t;     (** Covariance matrix *)
  omega: Tensor.t;     (** Hyper-covariance matrix *)
  sigma_sq: float;          (** Noise variance *)
}

type experiment_config = {
  dim: int;                (** Problem dimension p *)
  num_tasks: int;          (** Number of tasks L *)
  samples_per_task: int;   (** Samples per task n_l *)
  num_test_samples: int;   (** Number of test samples *)
  sigma_sq: float;         (** Noise variance *)
  num_runs: int;           (** Number of random runs *)
}

type riemannian_params = {
  max_iter: int;
  tol: float;
  step_size: float;
  beta: float;  (** momentum parameter *)
}

type correlation_params = {
  l0: int;           (** Number of tasks with full rank observations *)
  min_eigenval: float;  (** Minimum eigenvalue threshold *)
  sparsity: int;     (** Expected sparsity level *)
}

type convergence_result = {
  converged: bool;
  rate: float;
  error_bound: float;
}

type spectral_distribution = {
  support: float * float;  (** Support interval [a,b] *)
  density: float -> float; (** Density function *)
  stieltjes: Complex.t -> Complex.t; (** Stieltjes transform *)
}