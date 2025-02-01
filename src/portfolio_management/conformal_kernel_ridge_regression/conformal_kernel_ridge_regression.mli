open Torch

(** Numeric control for stability *)
module NumericControl : sig
  type precision_level = Single | Double | Mixed

  type stability_config = {
    precision: precision_level;
    epsilon: float;
    max_condition_number: float;
    pivot_threshold: float;
    decomposition_method: [`QR | `SVD | `Cholesky];
  }

  val default_config : stability_config
  val stable_solve : ?config:stability_config -> Tensor.t -> Tensor.t -> Tensor.t
end

(** Utility functions *)
module Utils : sig
  val sort_tensor : ?stable:bool -> Tensor.t -> Tensor.t * Tensor.t
  val eye : int -> Tensor.t
  val range : int -> Tensor.t
  val concat_vertical : Tensor.t -> Tensor.t -> Tensor.t
  val compute_quantile : Tensor.t -> float -> float
  val solve_system : ?config:NumericControl.stability_config -> Tensor.t -> Tensor.t -> Tensor.t
  val chunked_operation : chunk_size:int -> f:(Tensor.t -> 'a) -> Tensor.t -> 'a list
end

(** Kernel implementations *)
module Kernel : sig
  type kernel_type =
    | Gaussian
    | Linear
    | Polynomial
    | Laplacian
    | RationalQuadratic
    | Periodic

  module type S = sig
    type t
    val create : ?params:float array -> kernel_type -> t
    val compute : t -> Tensor.t -> Tensor.t -> Tensor.t
    val batch_compute : t -> Tensor.t -> Tensor.t -> Tensor.t
    val gradient : t -> Tensor.t -> Tensor.t -> Tensor.t
    val get_type : t -> kernel_type
    val to_device : t -> Device.t -> t
  end

  module KernelImpl : S
end

(** Kernel Ridge Regression *)
module KRR : sig
  type t = private {
    kernel: Kernel.KernelImpl.t;
    lambda: float;
    x_train: Tensor.t;
    y_train: Tensor.t;
    alpha: Tensor.t;
    gram_matrix: Tensor.t;
    cached_residuals: (Tensor.t * float) option;
  }

  val create : ?lambda:float -> ?kernel:Kernel.KernelImpl.t -> Tensor.t -> Tensor.t -> t
  val compute_in_sample_residuals : t -> Tensor.t
  val compute_loo_residuals : t -> Tensor.t
  val predict : t -> Tensor.t -> Tensor.t
  val log_likelihood : t -> float
end

(** Region computation utilities *)
module RegionComputation : sig
  type interval = {
    lower: float;
    upper: float;
    coverage: float;
    score: float;
  }

  type region_type =
    | EmptyRegion
    | FullRegion
    | SingleInterval of interval
    | UnionIntervals of interval list

  val compute_pvalue : Tensor.t -> float -> float
  val compute_regions : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> region_type list
  val compute_confidence_region : region_type list -> float -> region_type
end

(** Ridge Regression Confidence Machine *)
module RRCM : sig
  type t = private {
    krr: KRR.t;
    alpha: float;
  }

  val create : ?alpha:float -> KRR.t -> t
  val compute_confidence_region : t -> Tensor.t -> Tensor.t -> RegionComputation.region_type
end

(** Complete Two-sided Predictor *)
module CompleteTwoSidedPredictor : sig
  type t = private {
    krr: KRR.t;
    alpha: float;
  }

  val create : ?alpha:float -> KRR.t -> t
  val predict_region : t -> Tensor.t -> Tensor.t -> RegionComputation.region_type
end

(** Conformal validation *)
module ConformalValidation : sig
  type validation_result = {
    empirical_coverage: float;
    average_width: float;
    confidence_interval: float * float;
  }

  val compute_coverage : RegionComputation.region_type array -> float array -> validation_result
end

(** Parameter tuning *)
module ParameterTuning : sig
  type parameter_space = {
    lambda_range: float list;
    kernel_params: float list array;
    kernel_types: Kernel.kernel_type list;
  }

  type tuning_config = {
    max_iter: int;
    tolerance: float;
    parallel: bool;
    num_threads: int;
  }

  type tuning_result = {
    best_lambda: float;
    best_kernel_type: Kernel.kernel_type;
    best_kernel_params: float array;
    best_score: float;
    convergence_path: float array;
  }

  val grid_search : parameter_space -> tuning_config -> Tensor.t -> Tensor.t -> tuning_result
end