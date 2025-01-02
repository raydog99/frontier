open Torch

type time_series = {
  dimensions: int;
  sequence_length: int;
  data: Tensor.t
}

type model_config = {
  terminal_time: float;
  euler_steps: int;
  batch_size: int;
  learning_rate: float;
  max_epochs: int;
  initial_noise_std: float;
  inject_noise_std: float;
  potential_coeff: float;
}