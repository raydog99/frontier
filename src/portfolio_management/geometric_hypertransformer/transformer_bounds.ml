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

let compute_holder_bounds ~holder_params ~error =
  let open Float in
  let log2 x = log x /. log 2. in
  
  (* Width bound *)
  let width = match holder_params.activation_type with
    | `Singular ->
        ceil (
          (log2 (1. /. error.quantization) ** (2. /. holder_params.alpha)) *.
          float_of_int (holder_params.metric_capacity error.quantization) *.
          holder_params.domain_diameter ** holder_params.alpha
        )
    | `Smooth ->
        ceil (
          (1. +. log2 (1. /. error.encoding)) *.
          float_of_int holder_params.input_dim ** 
            (1. /. holder_params.alpha) *.
          float_of_int holder_params.output_dim
        )
    | `Classical ->
        ceil (
          float_of_int holder_params.input_dim *.
          float_of_int holder_params.output_dim *.
          log2 (1. /. (error.intrinsic +. error.quantization))
        )
  in

  let depth = match holder_params.activation_type with
    | `Singular -> 
        int_of_float width * int_of_float (log2 width) *
        int_of_float holder_params.domain_diameter
    | `Smooth ->
        int_of_float (
          ceil (log2 width *. 
               float_of_int holder_params.input_dim *.
               holder_params.alpha)
        )
    | `Classical ->
        int_of_float width + 
        int_of_float (log2 (1. /. error.encoding))
  in

  {
    function_space = {
      input_dim = holder_params.input_dim;
      output_dim = holder_params.output_dim;
      holder_exponent = holder_params.alpha;
      holder_constant = holder_params.constant;
    };
    approximation_error = error;
    complexity_bounds = {
      width = int_of_float width;
      depth;
      parameters = depth * int_of_float width * 
        int_of_float (log2 (1. /. error.quantization));
    };
  }

let compute_causal_bounds ~path_type ~holder_params ~error =
  let memory_size = match path_type with
    | `Exponential (c_star, _) ->
        int_of_float (ceil (
          log (1. /. error.memory) /. c_star
        ))
    | `Holder (alpha, c, _) ->
        int_of_float (ceil (
          (log (1. /. error.memory) /. alpha) *.
          c ** (1. /. alpha)
        ))
    | `Weighted (w, _) ->
        int_of_float (ceil (
          log (1. /. error.memory) *. w 0
        ))
  in

  let compression = match path_type with
    | `Exponential (c_star, eps) -> fun n ->
        4. *. eps ** (-1.) *. c_star *.
        sqrt holder_params.holder_constant *.
        exp (float_of_int n *. holder_params.holder_constant /. 2.)
    | `Holder (alpha, c, p) -> fun n ->
        4. *. error.quantization ** (-1.) *. c ** (1. /. p) *.
        float_of_int n ** (alpha /. 2.)
    | `Weighted (w, _) -> fun n ->
        4. *. error.quantization ** (-1.) *. w n *.
        holder_params.holder_constant
  in

  {
    memory_bound = {
      size = memory_size;
      horizon = 2 * memory_size;
      compression;
    };
    latent_bound = {
      dimension = memory_size * holder_params.output_dim;
      capacity = holder_params.metric_capacity error.quantization;
    };
    error_decomposition = {
      memory = error.memory;
      parameter = error.parameter;
      geometric = error.geometric;
    };
  }