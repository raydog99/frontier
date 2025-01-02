type observation = {
  x: float;
  y: float;
  z: float array;
}

type nuisance_estimates = {
  f_hat: float;
  m_hat: float;
}

type split_stats = {
  d_theta_sum: float;
  psi_sq_sum: float;
  n: int;
}

type node = {
  data_indices: int array;
  region: (float * float) array;
  split_var: int option;
  split_point: float option;
  left: node option;
  right: node option;
  stats: split_stats;
}

type fold = {
  train_indices: int array;
  test_indices: int array;
}

type estimation_stats = {
  theta_hat: float;
  sigma_hat: float;
  confidence_interval: float * float;
}

type tree = {
  root: node;
  weights: float array;
  tau1: float array;
  tau2: float array;
}

type forest = {
  trees: tree array;
  n_trees: int;
  folds: fold array;
}

type simulation_config = {
  model_type: Models.model_type;
  n_samples: int;
  n_dims: int;
  theta_true: float;
  noise_level: float;
}

type simulation_result = {
  theta_hat: float;
  bias: float;
  variance: float;
  mse: float;
  coverage: float;
}