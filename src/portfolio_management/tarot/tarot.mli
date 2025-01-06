open Torch

type dataset = {
  features: Tensor.t;
  labels: Tensor.t;
}

type gradient_features = {
  features: Tensor.t;
  checkpoints: int;
}

type transport_plan = {
  coupling: Tensor.t;
  cost: float;
}

module Checkpoint : sig
  type t = {
    model_state: (string * Tensor.t) list;
    optimizer_state: (string * Tensor.t) list;
    iteration: int;
    learning_rate: float;
  }

  val save : t -> string -> unit
  val load : string -> t
  val load_model : 'a -> t -> unit
  val create_ensemble : ?n_models:int -> 'a -> 'a list
end

module DataInfluence : sig
  type influence_config = {
    checkpoint_interval: int;
    max_checkpoints: int;
    lr_schedule: (int * float) list;
    gradient_norm_clip: float option;
    device: Device.t;
  }

  val default_config : influence_config
  val compute_gradient_checkpoints : ?config:influence_config -> 
    'a -> dataset -> Checkpoint.t list
  val compute_influence : dataset -> dataset -> Checkpoint.t list -> Tensor.t
end

module GradientFeatures : sig
  type embedding_config = {
    feature_dim: int;
    compression_ratio: float;
    batch_size: int;
    n_workers: int;
    device: Device.t;
    use_mixed_precision: bool;
    cache_dir: string option;
  }

  val default_config : embedding_config
  val embed_gradients : ?config:embedding_config -> 
    'a -> Checkpoint.t list -> dataset -> Tensor.t
end

module WhitenedFeatureDistance : sig
  type whitening_stats = {
    mean: Tensor.t;
    scale: Tensor.t;
    eigenvalues: Tensor.t;
    eigenvectors: Tensor.t;
  }

  val compute_whitening_stats : Tensor.t -> whitening_stats
  val apply_whitening : Tensor.t -> whitening_stats -> Tensor.t
  val update_stats : whitening_stats -> Tensor.t -> float -> whitening_stats
  val compute_wfd : Tensor.t -> Tensor.t -> float
end

module OptimalTransport : sig
  type selection_stats = {
    ot_distances: float array;
    selected_indices: int list;
    computation_time: float;
    memory_peak: int64;
  }

  type sinkhorn_params = {
    epsilon: float;
    max_iter: int;
    tolerance: float;
    stabilize_freq: int;
  }

  val default_sinkhorn_params : sinkhorn_params
  val sinkhorn : ?params:sinkhorn_params -> Tensor.t -> transport_plan
  val compute_ot_distance : dataset -> dataset -> epsilon:float -> float
end

module KFoldSelection : sig
  type fold_config = {
    k: int;
    validation_metric: Tensor.t -> Tensor.t -> float;
    min_fold_size: int;
    shuffle: bool;
    seed: int option;
  }

  val default_config : fold_config
  val select_with_validation : ?config:fold_config ->
    dataset -> dataset -> int list * OptimalTransport.selection_stats
end

module Selection : sig
  type selection_mode = FixedSize | OptimalTransport

  type coupling_params = {
    transport_weight: float;
    coverage_weight: float;
    diversity_weight: float;
    temperature: float;
  }

  type stopping_criteria = {
    max_iterations: int;
    min_improvement: float;
    patience: int;
    max_selections: int;
  }

  type selection_params = {
    mode: selection_mode;
    initial_size: int;
    growth_factor: float;
    coupling: coupling_params;
    stopping: stopping_criteria;
    device: Device.t;
  }

  val default_params : selection_params
  val select : ?params:selection_params -> 
    dataset -> dataset -> int list
end

module Training : sig
  type weighted_sampler = {
    indices: int array;
    weights: float array;
    alias: int array;
    prob: float array;
  }

  val create_weighted_sampler : float array -> weighted_sampler
  val create_weighted_dataset : dataset -> float array -> dataset
  val train_weighted : 'a -> 'b -> dataset -> float array -> 
    epochs:int -> batch_size:int -> unit
end