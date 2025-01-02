open Torch

type model =
  | LSTM of { num_layers: int; hidden_size: int; dropout: float }
  | Transformer of { num_layers: int; num_heads: int; d_model: int; dropout: float }
  | RandomForest of { num_trees: int; max_depth: int }
  | EnsembleModel of model list

val create_model : model -> int -> int -> Layer.t
val train : Layer.t -> (Tensor.t * Tensor.t) Torch_utils.Dataloader.t -> int -> float -> unit
val predict : Layer.t -> Tensor.t -> Tensor.t
val feature_importance : model -> float array
val cross_validate : Layer.t -> (Tensor.t * Tensor.t) Torch_utils.Dataloader.t -> int -> float list