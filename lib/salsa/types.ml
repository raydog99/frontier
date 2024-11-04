open Torch

type leverage_score = float
type sample_size = int
type probability = float
type matrix_dims = {rows: int; cols: int}

type arma_params = {
  ar_coefs: Tensor.t;
  ma_coefs: Tensor.t;
  white_noise_var: float;
  bic_score: float;
  std_errors: Tensor.t;
  residuals: Tensor.t;
}

type diagnostic_stats = {
  aic: float;
  bic: float;
  log_likelihood: float;
  durbin_watson: float;
  ljung_box_stat: float;
  residual_acf: Tensor.t;
}