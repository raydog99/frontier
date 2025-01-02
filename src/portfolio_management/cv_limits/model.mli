open Torch
open Types

module Model : sig
  type t = {
    model: ModelType.t;
    loss: LossType.t;
    optimizer: Optimizer.t;
  }

  val create : ModelType.t -> LossType.t -> float -> t
  val predict : t -> Tensor.t -> Tensor.t
  val evaluate : t -> Tensor.t -> Tensor.t -> float
  val train : t -> DataType.t -> epochs:int -> t
end

module LinearModel : sig
  val create : int -> int -> ModelType.t
end

module MLP : sig
  val create : int -> int -> int -> ModelType.t
end

module RandomForest : sig
  val create : int -> int -> int -> ModelType.t
end