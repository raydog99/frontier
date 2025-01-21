open Torch

(** Core types *)
type observation = Tensor.t
type parameters = Tensor.t
type mixture_indicator = int
type network_params = {
  input_dim: int;
  hidden_dims: int list;
  output_dim: int;
  learning_rate: float;
}

(** Core Network Module *)
module CoreNetwork : sig
  type t

  val create : int -> int list -> int -> t
  val forward : t -> Tensor.t -> Tensor.t
  val summarize : t -> Tensor.t -> Tensor.t
  val create_classifier : input_dim:int -> hidden_dims:int list -> num_components:int -> t
  val forward_classification : t -> Tensor.t -> Tensor.t -> Tensor.t
  val categorical_cross_entropy : Tensor.t -> Tensor.t -> Tensor.t
end

(** Distribution Module *)
module Distribution : sig
  type t = {
    loc: Tensor.t;
    scale: Tensor.t;
  }

  val gaussian : loc:Tensor.t -> scale:Tensor.t -> t
  val sample : t -> n:int -> Tensor.t
  val log_prob : t -> Tensor.t -> Tensor.t
end

(** Numerical Utilities *)
module NumericalUtils : sig
  type stability_config = {
    eps: float;
    max_value: float;
    min_value: float;
    log_space_threshold: float;
  }

  val default_config : stability_config
  val stabilize_log_sum_exp : Tensor.t -> dim:int -> stability_config -> Tensor.t
  val safe_log : Tensor.t -> stability_config -> Tensor.t
  val clip_gradients : Tensor.t list -> stability_config -> Tensor.t list
end

(** Memory Management *)
module MemoryManager : sig
  type buffer_config = {
    max_buffer_size: int;
    cleanup_threshold: float;
  }

  type t

  val create : buffer_config -> t
  val register_tensor : t -> string -> Tensor.t -> unit
  val get_tensor : t -> string -> Tensor.t option
  val cleanup : t -> unit
end

(** Inference Network *)
module InferenceNetwork : sig
  type inference_t

  val create_flow : int -> int list -> int -> inference_t
  val forward : inference_t -> Tensor.t -> Distribution.t
end

(** Normalizing Flow *)
module NormalizingFlow : sig
  type t

  val create : int -> int -> t
  val forward : t -> Tensor.t -> Tensor.t * Tensor.t
  val inverse : t -> Tensor.t -> Tensor.t
end

(** Forward Model *)
module ForwardModel : sig
  type t

  val create : num_components:int -> component_dim:int -> t
  val sample_prior : t -> batch_size:int -> parameters * mixture_indicator array
  val generate : t -> parameters -> Tensor.t
end

(** Mixture Dependencies *)
module MixtureDependencies : sig
  type dependency_type =
    | Independent
    | Markov
    | SemiMarkov of { duration_max: int }
    | FullyDependent

  type t

  val create_transition_model : dependency_type -> int -> t
  val sample_next_state : t -> int -> Tensor.t -> int
end

(** Advanced Sampling *)
module AdvancedSampling : sig
  type sampling_config = {
    num_chains: int;
    warmup_steps: int;
    adaptation_steps: int;
    target_acceptance: float;
  }

  module HMC : sig
    type state = {
      position: Tensor.t;
      momentum: Tensor.t;
      log_prob: float;
      grad: Tensor.t;
    }

    val leapfrog : 
      state -> 
      step_size:float -> 
      num_steps:int -> 
      potential_fn:(Tensor.t -> float) ->
      grad_fn:(Tensor.t -> float * Tensor.t) ->
      state

    val sample :
      init_state:state ->
      config:sampling_config ->
      potential_fn:(Tensor.t -> float) ->
      grad_fn:(Tensor.t -> float * Tensor.t) ->
      Tensor.t list
  end

  module BifurcatingSampler : sig
    type tree = {
      position: Tensor.t;
      momentum: Tensor.t;
      grad: Tensor.t;
      log_prob: float;
      depth: int;
    }

    val build_tree :
      HMC.state ->
      step_size:float ->
      potential_fn:(Tensor.t -> float) ->
      grad_fn:(Tensor.t -> float * Tensor.t) ->
      max_depth:int ->
      HMC.state * HMC.state
  end
end

(** Training *)
module Training : sig
  type training_config = {
    batch_size: int;
    num_epochs: int;
    learning_rate: float;
    weight_decay: float;
    gradient_clip_norm: float option;
    early_stopping_patience: int option;
    learning_rate_schedule: [`Step | `Exponential | `Cosine] option;
  }

  type training_state

  val train_model : ABMM.t -> training_config -> Tensor.t -> training_state
end

(** Model Validation *)
module ModelValidation : sig
  type validation_metrics = {
    train_divergence: float;
    validation_divergence: float;
    classification_accuracy: float;
    posterior_predictive_score: float;
  }

  val compute_metrics : ABMM.t -> Tensor.t -> Tensor.t -> validation_metrics
end

(** ABMM - Main Module *)
module ABMM : sig
  type t

  val create : 
    num_components:int -> 
    component_dim:int -> 
    summary_dim:int -> 
    hidden_dims:int list -> 
    t

  val train_step : t -> Tensor.t -> float
  val infer : t -> Tensor.t -> Distribution.t * Tensor.t
end

(** Integration *)
module Integration : sig
  type model_config = {
    num_components: int;
    input_dim: int;
    hidden_dims: int list;
    latent_dim: int;
    learning_rate: float;
    batch_size: int;
  }

  val create_and_train_model : 
    model_config -> 
    Tensor.t -> 
    ABMM.t * Training.training_state
end