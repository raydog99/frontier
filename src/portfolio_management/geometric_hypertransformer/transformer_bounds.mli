type holder_approximation = {
  function_space: {
    input_dim: int;
    output_dim: int;
    holder_exponent: float;
    holder_constant: float;
  };
  approximation_error: {
    intrinsic: float;
    quantization: float;
    encoding: float;
  };
  complexity_bounds: {
    width: int;
    depth: int;
    parameters: int;
  };
}

type causal_approximation = {
  memory_bound: {
    size: int;
    horizon: int;
    compression: float -> int -> float;
  };
  latent_bound: {
    dimension: int;
    capacity: int;
  };
  error_decomposition: {
    memory: float;
    parameter: float;
    geometric: float;
  };
}

val compute_holder_bounds:
  holder_params:holder_params ->
  error:approximation_error ->
  holder_approximation

val compute_causal_bounds:
  path_type:[`Exponential of float * float
           |`Holder of float * float * float
           |`Weighted of (int -> float) * float] ->
  holder_params:holder_params ->
  error:approximation_error ->
  causal_approximation