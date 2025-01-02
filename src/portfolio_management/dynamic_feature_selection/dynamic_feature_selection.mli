open Torch

module Types : sig
  type distribution =
    | Categorical of { probs: Tensor.t; num_classes: int }
    | Normal of { mu: Tensor.t; sigma: Tensor.t }
    | Bernoulli of { probs: Tensor.t }
    
  type feature_type =
    | Continuous
    | Categorical of int
    | Binary
    
  type feature_info = {
    feature_type: feature_type;
    name: string;
    index: int;
    dependencies: int list;
  }
  
  type model_config = {
    feature_dim: int;
    hidden_dims: int list;
    num_classes: int;
    feature_info: feature_info array;
    dropout_rate: float;
    use_batch_norm: bool;
    residual_connections: bool;
  }
end

module Dataset : sig
  type t = {
    features: Tensor.t;
    labels: Tensor.t;
    batch_size: int;
  }
  
  val create : Tensor.t -> Tensor.t -> int -> t
  val shuffle : t -> t
  val batches : t -> (Tensor.t * Tensor.t) list
  val split : t -> float -> t * t  (* train/val split ratio *)
end

module PolicyNetwork : sig
  type t
  
  val create : Types.model_config -> t
  val forward : t -> Tensor.t -> Tensor.t -> bool -> Tensor.t
  val parameters : t -> (string * Tensor.t) list
end

module PredictorNetwork : sig
  type t
  
  val create : Types.model_config -> t
  val forward : t -> Tensor.t -> Tensor.t -> bool -> Tensor.t
  val parameters : t -> (string * Tensor.t) list
  val loss : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
end

module CMI : sig
  val estimate : Tensor.t -> Tensor.t -> Tensor.t -> int -> float
  
  module Oracle : sig
    type t
    val create : Tensor.t -> Tensor.t -> t
    val estimate_cmi : t -> Tensor.t -> int -> float
  end
  
  module GreedyPolicy : sig
    type t
    val create : ?num_samples:int -> ?oracle:Oracle.t -> unit -> t
    val select_feature : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
  end
end

module Training : sig
  module LRScheduler : sig
    type t
    type scheduler_type =
      | StepLR of { step_size: int; gamma: float }
      | CosineAnnealingLR of { T_max: int; eta_min: float }
      | ReduceLROnPlateau of {
          factor: float;
          patience: int;
          min_lr: float;
          mutable best_score: float;
          mutable counter: int;
        }
        
    val create : ?scheduler_type:scheduler_type -> float -> t
    val step : t -> ?score:float -> unit -> unit
  end
  
  module Validation : sig
    type t
    val create : ?patience:int -> unit -> t
    val step : t -> 'a -> float -> unit
    val restore_best_model : t -> 'a -> unit
  end
  
  val train : Types.model_config -> Dataset.t -> Dataset.t -> 
             num_epochs:int -> initial_lr:float -> unit
end

module Metrics : sig
  type evaluation_metrics = {
    accuracy: float;
    mi_score: float;
    selection_stability: float;
    avg_num_features: float;
  }
  
  val accuracy : Tensor.t -> Tensor.t -> float
  val mutual_information : Tensor.t -> Tensor.t -> float
  val selection_stability : Tensor.t -> float
  val feature_importance : Tensor.t -> Tensor.t -> Tensor.t
end

module DFS : sig
  type t = {
    policy: PolicyNetwork.t;
    predictor: PredictorNetwork.t;
    config: Types.model_config;
  }
  
  val create : Types.model_config -> t
  val select_features : t -> Tensor.t -> int -> Tensor.t
  val predict : t -> Tensor.t -> Tensor.t -> Tensor.t
  val train : t -> Dataset.t -> Dataset.t -> num_epochs:int -> initial_lr:float -> unit
  val evaluate : t -> Dataset.t -> Metrics.evaluation_metrics
end

Let me continue with the implementation files. Would you like me to proceed?