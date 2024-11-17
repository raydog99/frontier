open Torch

type metrics = {
  mse: float;
  mae: float;
  relative_error: float;
  max_error: float;
}

type monitor_state = {
  epoch: int;
  metrics: metrics;
  best_model_state: Tensor.t list;
  early_stop_counter: int;
}

val create_monitor : unit -> monitor_state
val update : monitor_state -> WavKANNetwork.t -> 
             (Tensor.t * Tensor.t) -> monitor_state
val compute_metrics : Tensor.t -> Tensor.t -> metrics

module Training : sig
  type training_config = {
    batch_size: int;
    epochs: int;
    optimizer_config: Optimization.optimizer_config;
    regularizers: Regularization.regularizer_type list;
    early_stopping_patience: int;
  }

  val train_epoch : 
    WavKANNetwork.t -> 
    Optimization.t -> 
    ((Tensor.t -> Tensor.t -> unit) -> unit) -> 
    training_config -> 
    float

  val train : 
    WavKANNetwork.t -> 
    training_config -> 
    ((Tensor.t -> Tensor.t -> unit) -> unit) ->
    (unit -> (Tensor.t * Tensor.t)) ->
    Validation.monitor_state
end

module Adaptive : sig
  type adaptive_params = {
    scale: Tensor.t;
    translation: Tensor.t;
    frequency: Tensor.t option;
    shape: Tensor.t option;
  }

  type strategy =
    | Gradient
    | MetaLearning
    | Evolutionary

  type t

  val create : int -> int -> strategy -> t
  val forward : t -> Tensor.t -> Tensor.t
  val adapt : t -> t
  val get_adaptation_history : t -> adaptive_params list
end