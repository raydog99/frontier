type ('a, 'b) t

val create:
  base_transformer:('a, 'b) GeometricTransformer.t ->
  memory_size:int ->
  compression:TransformerBounds.CausalApproximation.causal_bounds ->
  ('a, 'b) t

val forward:
  ('a, 'b) t ->
  'a list ->
  'b list