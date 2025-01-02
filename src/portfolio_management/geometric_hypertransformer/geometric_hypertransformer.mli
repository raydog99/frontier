open Torch

type ('a, 'b) t

type memory_state = {
  parameters: Tensor.t;
  velocity: Tensor.t;
  momentum: float;
  time: int;
}

type network_params = {
  width: int;
  depth: int;
  head_dim: int;
  num_heads: int;
}

val create:
  base_transformer:('a, 'b) GeometricTransformer.t ->
  memory_size:int ->
  compression:TransformerBounds.CausalApproximation.causal_bounds ->
  ('a, 'b) t

val forward:
  ('a, 'b) t ->
  'a list ->
  'b list

val evolve_parameters:
  memory_state ->
  time:int ->
  compression_rate:(int -> float) ->
  memory_state

val compute_compression_rate:
  TransformerBounds.CausalApproximation.causal_bounds ->
  int ->
  float

(* Methods specific to the dynamic case *)
val estimate_memory_requirement:
  approximation_error:float ->
  holder_exponent:float ->
  metric_capacity:(float -> int) ->
  int

val verify_compression_bounds:
  ('a, 'b) t ->
  time:int ->
  bool