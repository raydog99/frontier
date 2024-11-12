open Torch
open Types

module FisherInfo : sig
  val compute_complete_data_fisher : model -> Tensor.t -> Tensor.t -> Tensor.t
  val compute_conditional_fisher : model -> Tensor.t -> Tensor.t -> Tensor.t
  val compute_mim : model -> Tensor.t -> Tensor.t -> Tensor.t
end

module Analysis : sig
  val compute_eigenvalues : Tensor.t -> Tensor.t
  val estimate_snr : model -> Tensor.t -> Tensor.t -> float
  val check_conditions : model -> Tensor.t -> Tensor.t -> theorem_conditions
  val compute_metrics : model -> Tensor.t -> Tensor.t -> convergence_metrics
  val verify_relative_smoothness : model -> Tensor.t -> Tensor.t -> float
  val analyze_snr : model -> Tensor.t -> Tensor.t -> float * float array * float
end