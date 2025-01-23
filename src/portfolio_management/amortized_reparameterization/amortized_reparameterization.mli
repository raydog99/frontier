open Torch

type stability_config = {
  eps: float;
  min_clamp: float;
  max_clamp: float;
  condition_threshold: float;
}

type sde_params = {
  drift: Tensor.t -> float -> Tensor.t;
  dispersion: float -> Tensor.t;
  initial_dist: unit -> Tensor.t * Tensor.t
}

type variational_params = {
  encoder_mean: Tensor.t list -> float -> Tensor.t;
  encoder_cov: Tensor.t list -> float -> Tensor.t;
  time_encoder: float -> Tensor.t
}

val default_stability_config : stability_config

(* Core tensor operations *)
val ensure_2d : Tensor.t -> Tensor.t
val batch_matmul : Tensor.t -> Tensor.t -> Tensor.t
val safe_inverse : ?config:stability_config -> Tensor.t -> Tensor.t
val safe_cholesky : ?config:stability_config -> Tensor.t -> Tensor.t
val safe_log : Tensor.t -> int -> Tensor.t
val safe_div : Tensor.t -> Tensor.t -> int -> Tensor.t
val safe_sqrt : Tensor.t -> int -> Tensor.t
val stable_softmax : Tensor.t -> int -> Tensor.t

(* Kronecker operations *)
val kron : Tensor.t -> Tensor.t -> Tensor.t
val kron_mv : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val kron_sum : Tensor.t -> Tensor.t -> Tensor.t

module SDECore : sig
  type sde_spec = {
    dim: int;
    latent_dim: int;
    hidden_dim: int;
    time_embedding_dim: int;
  }

  type t

  val make : sde_spec -> t
  val drift : t -> Tensor.t -> float -> Tensor.t
  val diffusion : t -> float -> Tensor.t
  val sample_initial : t -> int -> Tensor.t
end

module MGPCore : sig
  type t

  val make : int -> int -> t
  val compute_drift_params : t -> float -> int -> Tensor.t * Tensor.t
  val compute_derivatives : t -> Tensor.t -> Tensor.t -> float -> Tensor.t * Tensor.t
  val initialize : t -> int -> Tensor.t * Tensor.t
  val stable_covariance_update : t -> Tensor.t -> Tensor.t -> Tensor.t -> stability_config -> Tensor.t
end

module Amortization : sig
  type t

  val make : partition_size:int -> overlap:int -> latent_dim:int -> hidden_dim:int -> t
  val create_partitions : t -> Tensor.t list -> float list -> (Tensor.t list * float list) list
  val interpolate : t -> Tensor.t list -> float list -> float -> Tensor.t
end

module TemporalContext : sig
  type t

  val make : window_size:int -> stride:int -> kernel_size:int -> hidden_dim:int -> attention_heads:int -> t
  val attention_module : int -> int -> int -> Tensor.t -> Tensor.t
  val temporal_conv_module : int -> int -> Tensor.t -> Tensor.t
  val extract_neighbors : t -> 'a list -> int -> 'a list
  val process_sequence : t -> Tensor.t list -> Tensor.t list
end

module GradientEstimation : sig
  type t

  val make : outer_samples:int -> inner_samples:int -> stratification:bool -> batch_size:int -> t
  val estimate_gradients : t -> SDECore.t -> (Tensor.t -> float -> Tensor.t) -> Tensor.t list -> float * float -> Tensor.t list
end

module MonteCarlo : sig
  type t

  val make : parallel:bool -> batch_size:int -> num_samples:int -> t
  val sample_normal : Tensor.t -> Tensor.t -> Tensor.t
  val integrate : t -> (float -> Tensor.t) -> float * float -> int -> Tensor.t
  val importance_sampling : t -> (float -> Tensor.t) -> (unit -> float) -> float * float -> int -> Tensor.t
end

module ExtendedOptimization : sig
  type scheduler_type =
    | CosineAnnealing of {min_lr: float; max_lr: float; cycle_steps: int}
    | LinearWarmup of {warmup_steps: int; peak_lr: float}
    | CyclicalLR of {base_lr: float; max_lr: float; step_size: int}

  type t

  val make : scheduler:scheduler_type -> clip_grad_norm:float option -> 
            weight_decay:float -> ema_decay:float option -> t
  val optimization_step : t -> Var_store.t -> int -> unit
end

module Trainer : sig
  type config = {
    epochs: int;
    batch_size: int;
    partition_size: int;
    optimizer: ExtendedOptimization.t;
    grad_estimator: GradientEstimation.t;
    checkpoint_interval: int;
    early_stopping_patience: int;
  }

  val train : config -> LatentSDE.t -> Trajectory.t list -> unit
end

module BatchProcessing : sig
  type t

  val make : batch_size:int -> shuffle:bool -> drop_last:bool -> t
  val create_batches : t -> Tensor.t list * float list * int list -> 
                     (Tensor.t list * float list * int list) list
end

module DataAugmentation : sig
  type augmentation_type =
    | TimeShift of float
    | TimeScale of float
    | AdditiveSmoothNoise of float
    | TemporalMixup of float

  type t

  val make : augmentations:augmentation_type list -> probability:float -> t
  val apply_augmentation : t -> Tensor.t list * float list * int list -> 
                          Tensor.t list * float list * int list
end

module Trajectory : sig
  type t = {
    data: Tensor.t list;
    times: float list;
    batch_indices: int list;
  }

  val make : Tensor.t list -> float list -> int list -> t
  val to_batches : t -> int -> (Tensor.t list * float list * int list) list
  val collate : t list -> t
end

module Initialization : sig
  type init_method =
    | Xavier
    | KaimingNormal
    | Orthogonal
    | Custom of (Tensor.t -> Tensor.t)

  val init_weights : init_method -> Tensor.t -> Tensor.t
  val init_network : init_method -> Var_store.t -> unit
end

module LatentSDE : sig
  type model_config = {
    dim: int;
    latent_dim: int;
    hidden_dim: int;
    partition_size: int;
    num_samples: int;
  }

  type t

  val make : config:model_config -> device:Device.t -> t
  val forward : t -> Tensor.t -> float -> Tensor.t * Tensor.t * Tensor.t
  val compute_loss : t -> Tensor.t -> float -> Tensor.t
  val train : t -> model_config -> Trajectory.t list -> unit
end

module Util : sig
  val tensor_to_list : Tensor.t -> float list
  val list_to_tensor : float list -> Tensor.t
  val split_list : 'a list -> int -> 'a list list
  val pad_list : 'a list -> int -> 'a -> 'a list
  val sliding_window : 'a list -> int -> int -> 'a list list
end