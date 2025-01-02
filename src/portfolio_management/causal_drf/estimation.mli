open Torch

type ckte_estimate = {
  point_estimate: Tensor.t;
  standard_error: Tensor.t;
  kernel_type: kernel_type;
  effective_sample_size: float;
}

type confidence_band = {
  lower: Tensor.t;
  upper: Tensor.t;
  pointwise_coverage: float;
  uniform_coverage: float;
  band_width: float;
}

val estimate_ckte : sample -> forest -> Tensor.t -> ckte_estimate
val construct_uniform_bands : sample -> forest -> Tensor.t -> float -> confidence_band
val calculate_witness_function : sample -> kernel_type -> Tensor.t -> Tensor.t -> float
val estimate_ckte_with_samples : sample -> forest -> Tensor.t -> half_sample array -> ckte_estimate
val get_critical_value : float -> float
val calculate_standard_error : sample -> float array -> int array -> int array -> Tensor.t -> Tensor.t -> Tensor.t
val verify_coverage : confidence_band -> Tensor.t -> float array -> bool