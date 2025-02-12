open Torch

(** Configuration type for divergence parameters *)
type divergence_config = {
  mu: float;          (** Smoothness parameter *)
  beta1: float;       (** First regularization parameter *)
  beta2: float;       (** Second regularization parameter *)
}

(** Configuration type for model parameters *)
type model_config = {
  input_dim: int;     (** Input dimension *)
  rank: int;          (** Rank of factorization *)
  max_order: int;     (** Maximum order of feature interactions *)
  learning_rate: float;
  divergence: divergence_config;
}

(** Numerical stability utilities *)
val epsilon : float
val stable_log : float -> float
val stable_exp : float -> float
val clip : Tensor.t -> float -> Tensor.t
val stable_sum : float array -> float

(** Divergence functions *)
module Divergence : sig
  type t =
    | SquaredDivergence
    | LogisticDivergence
    | HingeDivergence

  val compute : t -> float -> float -> float
  val gradient : t -> float -> float -> float
end

(** Parameter initialization *)
module Init : sig
  type method_t =
    | Xavier of float
    | HeNormal of float
    | HeUniform of float
    | LeCunNormal of float
    | LeCunUniform of float
    | Orthogonal of float
    | Zeros
    | Constant of float

  val init : method_t -> int list -> Tensor.t
end

(** Dynamic programming *)
module DynamicProgramming : sig
  val compute_anova_table : Tensor.t -> Tensor.t -> int -> Tensor.t
  val compute_gradient_table : Tensor.t -> Tensor.t -> Tensor.t -> int -> Tensor.t
end

(** ANOVA kernel *)
module Anova : sig
  type t = private {
    p: Tensor.t;
    order: int;
    cache: (int, float array) Hashtbl.t;
  }

  val create : model_config -> t
  val eval_kernel : t -> Tensor.t -> float
  val grad_kernel : t -> Tensor.t -> Tensor.t
end

(** Higher-order factorization machine *)
module HOFM : sig
  type t = private {
    w: Tensor.t;
    p: Tensor.t array;
    config: model_config;
  }

  val create : model_config -> t
  val predict : t -> Tensor.t -> float
  val grad_kernel : t -> Tensor.t -> Tensor.t -> int -> Tensor.t
end

(** Shared parameters *)
module SharedParameters : sig
  type t = private {
    w: Tensor.t;
    shared_factors: Tensor.t;
    degree_weights: Tensor.t;
    dummy_features: Tensor.t;
    config: model_config;
    mutable feature_map: Tensor.t option;
  }

  val create : model_config -> t
  val compute_feature_mapping : t -> Tensor.t -> Tensor.t
  val predict : t -> Tensor.t -> float
  val compute_gradient : t -> Tensor.t -> float -> float -> 
    (Tensor.t * Tensor.t * Tensor.t)
end

(** Optimizer *)
module Optimizer : sig
  type t =
    | SGD of float
    | Adam of {
        alpha: float;
        beta1: float;
        beta2: float;
        epsilon: float;
        mutable m: Tensor.t;
        mutable v: Tensor.t;
        mutable t: int;
      }

  val create : [ `SGD of float | `Adam of float * float * float * float ] -> t
  val step : t -> Tensor.t -> Tensor.t -> Tensor.t
end