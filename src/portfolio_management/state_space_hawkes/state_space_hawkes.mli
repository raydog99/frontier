open Torch

module ComplexOps : sig
  type t = {re: Tensor.t; im: Tensor.t}
  
  val eps : float
  val complex_mul : t -> t -> t
  val complex_add : t -> t -> t
  val complex_exp : t -> t
  val to_complex : Tensor.t -> t
  val complex_matrix_mul : t -> t -> t
  val safe_log : Tensor.t -> Tensor.t
  val safe_div : Tensor.t -> Tensor.t -> Tensor.t
end

type event = {
  time: float;
  mark: int;
  index: int;
}

type history = {
  events: event array;
  times: Tensor.t;
  marks: Tensor.t;
  count: int;
}

type model_config = {
  hidden_dim: int;
  mark_dim: int;
  embedding_dim: int;
  num_layers: int;
  use_input_dependent: bool;
  max_memory: int;
}

type layer_state = {
  x: ComplexOps.t;
  u: Tensor.t;
  y: Tensor.t;
  time: float;
}

val empty_layer_state : layer_state
val stabilize_eigenvalues : Tensor.t -> ComplexOps.t
val compute_condition_number : Tensor.t -> Tensor.t -> float
val qr_decomposition : Tensor.t -> Tensor.t * Tensor.t

module ComplexEigenSystem : sig
  type eigen_system = {
    eigenvalues: ComplexOps.t array;
    v: ComplexOps.t;
    v_inv: ComplexOps.t;
    condition_number: float;
  }
  
  val balance_matrix : ComplexOps.t -> ComplexOps.t * Tensor.t
  val compute_eigensystem : Tensor.t -> eigen_system
end

module SJDE : sig
  type jump_process = {
    intensity: Tensor.t;
    mark_effect: Tensor.t;
    time: float;
  }

  val compute_jump_effect : model_config -> layer_state -> jump_process -> Tensor.t
  val integrate_jumps : model_config -> layer_state -> jump_process array -> float -> Tensor.t
  val solve_sjde : LLH_Layer.t -> layer_state -> jump_process array -> float -> layer_state
end

module StateManagement : sig
  type compression_config = {
    compression_ratio: int;
    max_memory: int;
    interpolation_points: int;
  }

  type state_cache = {
    states: (int, layer_state) Hashtbl.t;
    times: float array;
    max_size: int;
  }

  val create_cache : int -> state_cache
  val compress_state : layer_state -> compression_config -> layer_state
  val interpolate_states : layer_state -> layer_state -> float -> layer_state
  val get_state_at_time : state_cache -> float -> layer_state
  val update_cache : state_cache -> layer_state -> unit
end

module LayerNorm : sig
  type t = {
    gamma: Tensor.t;
    beta: Tensor.t;
    eps: float;
  }

  val create : int -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module MarkEmbedding : sig
  type t = {
    weight: Tensor.t;
    dim: int;
    num_marks: int;
  }

  val create : int -> int -> t
  val forward : t -> int -> Tensor.t
  val compute_attention : t -> Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t
end

module LLH_Layer : sig
  type t = {
    a: ComplexOps.t;
    b: Tensor.t;
    e: Tensor.t;
    c: Tensor.t;
    d: Tensor.t;
    eigenvals: ComplexOps.t;
    v: ComplexOps.t;
    v_inv: ComplexOps.t;
    config: model_config;
  }

  val create : model_config -> t
  val discretize : t -> float -> t
  val forward : t -> layer_state -> Tensor.t -> layer_state
  val forward_with_jumps : t -> layer_state -> float -> int -> layer_state
end

module LHP : sig
  type t = {
    layers: LLH_Layer.t array;
    mark_embedding: MarkEmbedding.t;
    layer_norms: LayerNorm.t array;
    output_projection: Tensor.t;
    output_bias: Tensor.t;
    scale: Tensor.t;
    config: model_config;
    state_cache: StateManagement.state_cache;
  }

  val create : model_config -> t
  val forward_layer : t -> int -> layer_state -> float -> layer_state
  val compute_intensity : t -> layer_state -> float -> Tensor.t
end

module Training : sig
  type training_config = {
    batch_size: int;
    learning_rate: float;
    max_epochs: int;
    patience: int;
    grad_clip: float;
    weight_decay: float;
    scheduler_factor: float;
    min_lr: float;
    num_monte_carlo: int;
  }

  type optimizer_state

  val create_optimizer_state : (string * Tensor.t) list -> optimizer_state
  val collect_parameters : LHP.t -> (string * Tensor.t) list
  val adam_update : (string * Tensor.t) list -> optimizer_state -> float -> optimizer_state
  val clip_gradients : (string * Tensor.t) list -> float -> unit
  val compute_loss : LHP.t -> history -> training_config -> float
end

module BatchProcessor : sig
  type batch = {
    states: layer_state array array;
    times: float array;
    marks: int array;
    masks: Tensor.t;
    memory_config: StateManagement.compression_config;
  }

  val create_batch : int -> model_config -> batch
  val create_batches : history -> Training.training_config -> batch array
  val process_batch : LHP.t -> batch -> layer_state array array
end

module Evaluation : sig
  type metrics = {
    log_likelihood: float;
    mean_intensity: float;
    mark_distribution: Tensor.t;
  }

  type prediction = {
    time: float;
    mark: int;
    intensity: Tensor.t;
    probability: float;
  }

  val compute_metrics : LHP.t -> history -> metrics
  val predict_next_event : LHP.t -> history -> float -> prediction
  val sample_categorical : Tensor.t -> int
end

module Visualization : sig
  type plot_config = {
    num_points: int;
    time_range: float * float;
    mark_colors: string array;
  }

  val create_intensity_plot_data : LHP.t -> history -> plot_config -> (float * Tensor.t) array
  val create_mark_embedding_viz : LHP.t -> float array array
end

module Logging : sig
  type log_entry = {
    epoch: int;
    train_loss: float;
    val_loss: float option;
    learning_rate: float;
    metrics: Evaluation.metrics;
    timestamp: float;
  }

  type training_log = {
    model_config: model_config;
    entries: log_entry list;
    total_time: float;
  }

  val create_log_entry : int -> float -> float option -> float -> Evaluation.metrics -> log_entry
  val save_log : training_log -> string -> unit
  val print_progress : log_entry -> unit
  val save_model : LHP.t -> string -> unit
  val load_model : string -> LHP.t
end