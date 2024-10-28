open Torch

type projection_config = {
  lambda: float;         (** Tuning parameter λ *)
  lambda_x: float;       (** Tuning for debiasing *)
  an: float;            (** Prior constant *)
  iterations: int;       (** Number of MCMC iterations *)
  burn_in: int;         (** Number of burn-in iterations *)
}

type posterior_sample = {
  theta: Tensor.t;
  theta_star: Tensor.t;
  theta_debiased: Tensor.t;
  sigma: float;
}

type distributed_data = {
  xtx: Tensor.t;  (** X'X *)
  xty: Tensor.t;  (** X'y *)
  n: int;               (** Total sample size *)
  p: int;               (** Number of features *)
}

type model_metrics = {
  tpr: float;    (** True Positive Rate *)
  fdp: float;    (** False Discovery Proportion *)
  mcc: float;    (** Matthews Correlation Coefficient *)
}

type credible_interval = {
  lower: float;
  upper: float;
  median: float;
  coverage: float;
}

type credible_ellipsoid = {
  center: Tensor.t;
  precision: Tensor.t;
  radius: float;
}

type convergence_diagnostics = {
  potential_scale_reduction: float;
  effective_sample_size: float;
  autocorrelation: float array;
}

type hypothesis_test = {
  parameter_idx: int;
  null_value: float;
  p_value: float;
  test_statistic: float;
  ci_lower: float;
  ci_upper: float;
}

type joint_test = {
  indices: int list;
  null_values: float list;
  test_statistic: float;
  p_value: float;
}

type adaptive_config = {
  base_config: projection_config;
  adaptation_period: int;
  target_acceptance: float;
  step_size: float ref;
}

type model_comparison = {
  dic: float;        (** Deviance Information Criterion *)
  waic: float;       (** Widely Applicable Information Criterion *)
  lpd: float;        (** Log Predictive Density *)
  p_eff: float;      (** Effective Number of Parameters *)
}

type sample_storage = {
  theta_star: float array;
  sigma: float;
  iteration: int;
  chain_id: int;
  timestamp: float;
}

type prediction = {
  mean: Tensor.t;
  lower: Tensor.t;
  upper: Tensor.t;
  std: Tensor.t;
}

type cv_fold = {
  train_x: Tensor.t;
  train_y: Tensor.t;
  test_x: Tensor.t;
  test_y: Tensor.t;
}

type cv_result = {
  lambda: float;
  mse: float;
  mae: float;
  r2: float;
}