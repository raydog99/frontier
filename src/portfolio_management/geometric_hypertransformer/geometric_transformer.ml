open Torch

type ('a, 'b) t = {
  encoder: Tensor.t -> Tensor.t;
  attention: GeometricAttention.t;
  decoder: Tensor.t -> 'b;
  bounds: complexity_bounds;
  qas_space: (module QASSpace.QAS_SPACE);
}

type complexity_bounds = {
  width: int;
  depth: int;
  parameters: int;
  holder_exponent: float;
  compression_rate: int -> float -> float;
}

type approximation_error = {
  intrinsic: float;
  quantization: float;
  encoding: float;
  total: float;
}

let compute_bounds ~holder_params ~error =
  let open Float in
  let log2 x = log x /. log 2. in
  
  let width = match holder_params.activation_type with
    | `Singular ->
        ceil (
          (log2 (1. /. error.quantization) ** (2. /. holder_params.alpha)) *.
          holder_params.metric_capacity *.
          holder_params.domain_diameter ** holder_params.alpha
        )
    | `Smooth ->
        ceil (
          (1. +. log2 (1. /. error.encoding)) *.
          holder_params.input_dim ** (1. /. holder_params.alpha) *.
          holder_params.output_dim
        )
    | `Classical ->
        ceil (
          holder_params.input_dim *.
          holder_params.output_dim *.
          log2 (1. /. error.total)
        )
  in

  let depth = match holder_params.activation_type with
    | `Singular -> 
        int_of_float width * int_of_float (log2 width) *
        int_of_float holder_params.domain_diameter
    | `Smooth ->
        int_of_float (ceil (log2 width *. 
                           float_of_int holder_params.input_dim *.
                           holder_params.alpha))
    | `Classical ->
        int_of_float width + 
        int_of_float (log2 (1. /. error.encoding))
  in

  let compression_rate = function
    | `ExponentialPath (c_star, epsilon) -> fun n ->
        4. *. epsilon ** (-1.) *. 
        sqrt holder_params.holder_constant *.
        exp (float_of_int n *. holder_params.holder_constant /. 2.)
    | `HolderPath (alpha, c, p) -> fun n ->
        4. *. error.quantization ** (-1.) *. c ** (1. /. p) *.
        float_of_int n ** (alpha /. 2.)
    | `WeightedPath w -> fun n ->
        4. *. error.quantization ** (-1.) *. w n *.
        holder_params.holder_constant
  in

  {
    width = int_of_float width;
    depth;
    parameters = depth * int_of_float width * 
      int_of_float (log2 (1. /. error.quantization));
    holder_exponent = holder_params.alpha;
    compression_rate;
  }

let create ~input_dim ~qas_space ~holder_params ~error =
  let bounds = compute_bounds ~holder_params ~error in
  
  let encoder = FeedForward.create 
    ~input_dim 
    ~hidden_dims:[bounds.width]
    ~output_dim:bounds.width
    ~activation_type:holder_params.activation_type in
    
  let attention = GeometricAttention.create
    ~num_heads:(bounds.width / 8)
    ~head_dim:(bounds.width / 8)
    ~qas_space in
    
  let decoder = FeedForward.create
    ~input_dim:bounds.width
    ~hidden_dims:[bounds.width]
    ~output_dim:holder_params.output_dim
    ~activation_type:holder_params.activation_type in

  {
    encoder = (fun x -> FeedForward.forward encoder x);
    attention;
    decoder = (fun x -> FeedForward.forward decoder x);
    bounds;
    qas_space;
  }

let forward ?params t input =
  let module QAS = (val t.qas_space) in
  let encoded = t.encoder input in
  let attention_output = 
    GeometricAttention.forward t.attention encoded in
  let decoded = t.decoder attention_output in
  QAS.quantize 
    (QAS.quantization_modulus t.bounds.quantization)
    decoded