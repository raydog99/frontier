open Torch
open Score_network
open Sde

type config = {
  noise_level: float;
  num_steps: int;
  num_samples: int;
  corrector_steps: int;
  tv_weight: float;
  fourier_weight: float;
  fourier_threshold: float;
}

val denoise : SDE -> Score_network.t -> Tensor.t -> config -> Tensor.t
val denoise_time_series : SDE -> Score_network.t -> Tensor.t -> config -> Tensor.t