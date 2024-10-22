open Torch

module type MODEL_CONFIG = sig
  val input_dim: int
  val qas_space: (module QAS_SPACE)
  val holder_params: HolderApproximation.params
  val approximation_error: TransformerBounds.approximation_error
end

module MakeModel : functor (Config : MODEL_CONFIG) -> sig
  val create_model: unit -> 
    (Tensor.t, Tensor.t) GeometricHypertransformer.t
end

module Training : sig
  type training_config = {
    learning_rate: float;
    batch_size: int;
    num_epochs: int;
    momentum: float;
  }

  val train_step:
    ('a, 'b) GeometricHypertransformer.t ->
    ('a * 'b) list ->
    training_config ->
    ('a, 'b) GeometricHypertransformer.t * float

  val evaluate:
    ('a, 'b) GeometricHypertransformer.t ->
    ('a * 'b) list ->
    float
end