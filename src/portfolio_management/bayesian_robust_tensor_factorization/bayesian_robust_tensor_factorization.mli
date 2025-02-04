open Torch

(* Core tensor operations *)
val hadamard_product_n : Tensor.t array -> Tensor.t
val kronecker_product : Tensor.t -> Tensor.t -> Tensor.t
val khatri_rao_product : Tensor.t -> Tensor.t -> Tensor.t
val khatri_rao_product_list : Tensor.t list -> Tensor.t
val generalized_inner_product : Tensor.t list -> Tensor.t
val matricize : Tensor.t -> int -> Tensor.t
val tensor_product : Tensor.t array -> Tensor.t
val cp_reconstruction : Tensor.t array -> Tensor.t

(* Numerical utilities *)
val stabilize_precision : Tensor.t -> Tensor.t
val safe_log : Tensor.t -> Tensor.t  
val safe_div : Tensor.t -> Tensor.t -> Tensor.t
val safe_sqrt : Tensor.t -> Tensor.t

(* Gamma distribution *)
module Gamma : sig
  type t = {
    shape: float;
    rate: float;
  }

  val mean : t -> float
  val variance : t -> float
  val log_expectation : t -> float
  val entropy : t -> float
  val kl_divergence : t -> t -> float
end

(* Multivariate normal distribution *)
module MultivariateNormal : sig
  type t = {
    mean: Tensor.t;
    covariance: Tensor.t;
    precision: Tensor.t option;
  }

  val create : Tensor.t -> Tensor.t -> t
  val create_with_precision : Tensor.t -> Tensor.t -> t
  val log_prob : t -> Tensor.t -> float
  val kl_divergence : t -> t -> float
end

(* Model configuration *)
module ModelConfig : sig
  type t = {
    order: int;
    dimensions: int array;
    rank: int;
    noise_precision: float;
    max_iter: int;
    tolerance: float;
  }

  val create : int -> int array -> int -> float -> int -> float -> t
end

(* Model state *)
module ModelState : sig
  type t = {
    factor_means: Tensor.t array;
    factor_covs: Tensor.t array;
    lambda_shape: float;
    lambda_rate: Tensor.t;
    sparse_mean: Tensor.t;
    sparse_precision: Tensor.t;
    gamma_shape: float;
    gamma_rate: Tensor.t;
    tau_shape: float;
    tau_rate: float;
    elbo: float;
    iteration: int;
  }

  val create : ModelConfig.t -> t
end

(* Posterior updates *)
module PosteriorUpdates : sig
  val update_factor_posterior : ModelState.t -> Tensor.t -> Tensor.t -> int -> Tensor.t * Tensor.t
  val update_sparse_posterior : ModelState.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
  val update_lambda_posterior : ModelState.t -> float * Tensor.t
  val update_gamma_posterior : ModelState.t -> float * Tensor.t
  val update_tau_posterior : ModelState.t -> Tensor.t -> Tensor.t -> float * float
end

(* Model evidence *)
module ModelEvidence : sig
  val compute_expected_log_likelihood : ModelState.t -> Tensor.t -> Tensor.t -> float
  val compute_kl_factors : ModelState.t -> float
  val compute_kl_lambda : ModelState.t -> float
  val compute_kl_gamma : ModelState.t -> float
  val compute_kl_tau : ModelState.t -> float
  val compute_elbo : ModelState.t -> Tensor.t -> Tensor.t -> float
end

(* Inference *)
module Inference : sig
  val update_state : ModelState.t -> Tensor.t -> Tensor.t -> ModelState.t
  val check_convergence : ModelState.t -> float -> ModelConfig.t -> bool
  val fit : Tensor.t -> Tensor.t -> ModelConfig.t -> ModelState.t
end

(* Prediction *)
module Prediction : sig
  type prediction = {
    mean: Tensor.t;
    variance: Tensor.t;
    samples: Tensor.t array option;
  }

  val compute_mean_prediction : ModelState.t -> Tensor.t
  val compute_variance : ModelState.t -> Tensor.t
  val sample_predictive : ModelState.t -> int -> Tensor.t array option
  val predict : ModelState.t -> ?n_samples:int -> unit -> prediction
end