open Torch

type data_batch = {
  x: Tensor.t;
  y: Tensor.t;
  output_idx: int array;
  input_idx: int array;
}

type training_config = {
  num_epochs: int;
  batch_size: int;
  learning_rate: float;
  num_samples: int;
  num_data: int;
  num_outputs: int;
  jitter: float;
  diagonalize: bool;
}

module MatrixOps : sig
  type decomposition = {
    eigvals: Tensor.t;
    eigvecs: Tensor.t;
  }

  val decompose_stable : Tensor.t -> decomposition
  val cholesky_stable : Tensor.t -> Tensor.t
  val condition_number : Tensor.t -> Tensor.t
  val trace_product : Tensor.t -> Tensor.t -> Tensor.t
  val efficient_trace : Tensor.t -> Tensor.t
  val kron : Tensor.t -> Tensor.t -> Tensor.t
end

module type Kernel = sig
  type t
  val create : ?ard:bool -> int -> t
  val forward : t -> Tensor.t -> Tensor.t -> Tensor.t
  val diag : t -> Tensor.t -> Tensor.t
  val get_params : t -> Tensor.t list
  val set_params : t -> Tensor.t list -> t
  val num_params : t -> int
end

module SE_ARD_Kernel : Kernel
module ExponentialKernel : Kernel

module type Likelihood = sig
  type t
  val create : ?noise_var:float -> unit -> t
  val log_prob : t -> data_batch -> Tensor.t -> Tensor.t
  val predict : t -> Tensor.t -> Tensor.t -> (Tensor.t * Tensor.t)
  val get_params : t -> Tensor.t list
end

module GaussianLikelihood : Likelihood

module VariationalDist : sig
  type t
  val create : int -> t
  val sample : t -> num_samples:int -> Tensor.t
  val kl_divergence : t -> Tensor.t
  val get_params : t -> Tensor.t list
  val set_params : t -> Tensor.t list -> t
end

module Diagonalize : sig
  type t = private {
    mean: Tensor.t;
    scale: Tensor.t;
    condition_threshold: float;
  }
  val transform : t -> Tensor.t -> Tensor.t
  val inverse_transform : t -> Tensor.t -> Tensor.t
end

module ELBO : sig
  type t = private {
    num_data: int;
    num_outputs: int;
    jitter: float;
  }
  val create : num_data:int -> num_outputs:int -> ?jitter:float -> unit -> t
  val compute_kl_gaussian : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
  val compute_expected_ll : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
end

module StochasticOptimizer : sig
  type t
  val create : ?beta1:float -> ?beta2:float -> ?epsilon:float -> 
              learning_rate:float -> Tensor.t list -> t
  val step : t -> Tensor.t list -> t
end

module NaturalGradient : sig
  type t
  val create : ?learning_rate:float -> ?momentum:float -> ?damping:float -> unit -> t
  val compute_fisher_factor : Tensor.t list -> Tensor.t list
  val update_fisher_factors : t -> Tensor.t list -> t
  val compute_natural_gradient : t -> Tensor.t list -> Tensor.t list
end

module ParameterConstraints : sig
  type constraint_type =
    | Positive
    | Bounded of {lower: float; upper: float}
    | Probability
    | Custom of (float -> float)

  type t

  val create : (string * constraint_type) list -> t
  val apply_transform : t -> string -> float -> float
  val apply_inv_transform : t -> string -> float -> float
  val check_constraints : t -> (string * float) list -> bool
end

module LMC : sig
  type t

  val create : num_outputs:int -> num_latent:int -> input_dim:int -> t
  val compute_kernel : t -> Tensor.t -> Tensor.t -> Tensor.t
  val predict : t -> Tensor.t -> Tensor.t * Tensor.t
end

module LV_MOGP : sig
  type t

  val create : num_outputs:int -> latent_dim:int -> input_dim:int -> 
              num_inducing:int -> t
  val compute_elbo : t -> data_batch -> Tensor.t
  val predict : t -> Tensor.t -> Tensor.t * Tensor.t
end

module GS_LVMOGP : sig
  type t

  val create : num_outputs:int -> num_latents:int -> latent_dim:int -> 
              input_dim:int -> num_inducing_x:int -> num_inducing_h:int -> 
              ?config:training_config -> unit -> t
  val train : t -> Tensor.t -> Tensor.t -> unit
  val predict : t -> Tensor.t -> Tensor.t list * Tensor.t list
  val sample_predictive : t -> Tensor.t -> int -> Tensor.t list
  val get_parameters : t -> Tensor.t list
  val set_parameters : t -> Tensor.t list -> t
end

module ModelUtils : sig
  val preprocess_data : Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
  val cross_validate : GS_LVMOGP.t -> Tensor.t -> Tensor.t -> int -> float list
  val optimize_hyperparams : GS_LVMOGP.t -> Tensor.t -> Tensor.t -> 
                           (string * float) list
  val monitor_performance : GS_LVMOGP.t -> Tensor.t -> Tensor.t -> unit
  val save_model : GS_LVMOGP.t -> string -> unit
  val load_model : string -> GS_LVMOGP.t
end