open Torch

(** Safely divide two tensors, avoiding division by zero. *)
val safe_div : Tensor.t -> Tensor.t -> Tensor.t

(** Clip a tensor to be within a specified range. *)
val clip_tensor : Tensor.t -> float -> float -> Tensor.t

(** Sample with replacement from a list. *)
val sample_with_replacement : 'a list -> int -> 'a list

(** Transpose a list of lists. *)
val transpose : 'a list list -> 'a list list

(** Perform linear regression on two tensors. *)
val linear_regression : Tensor.t -> Tensor.t -> Tensor.t * Tensor.t

(** Perform Bayesian optimization. *)
val bayesian_optimization : (('a * float) list -> float) -> ('a * float * float) list -> ('a * float) list * float

(** Train a process on given data. *)
val train_process : Non_markovian_jump.Process.t -> (float * float) list -> Non_markovian_jump.Process.t

(** Compute validation error for a process. *)
val compute_validation_error : Non_markovian_jump.Process.t -> (float * float) list -> float

(** Update process parameters. *)
val update_process_parameters : Non_markovian_jump.Process.t -> Tensor.t -> Non_markovian_jump.Process.t

(** Compute log-likelihood for a process. *)
val compute_log_likelihood : Non_markovian_jump.Process.t -> (float * float) list -> float

(** Perform Bayesian model selection. *)
val bayesian_model_selection : Non_markovian_jump.Process.t list -> (float * float) list -> (Non_markovian_jump.Process.t * float) list

(** Gaussian Process module for Bayesian optimization. *)
module GaussianProcess : sig
  type t

  val create : float list list -> float list -> t
  val predict : t -> float list -> float * float
  val next_sample : t -> ('a * float * float) list -> float list
end

(** Normal distribution functions. *)
module Normal : sig
  val pdf : float -> float
  val cdf : float -> float
end