open Torch

type point = Tensor.t
type vector = Tensor.t  
type matrix = Tensor.t

module Numerical : sig
  type integration_method = 
    | Trapezoidal
    | Simpson
    | GaussLegendre of int

  val adaptive_integrate : (float -> float) -> float -> float -> float -> int -> Tensor.t
end

module KDTree : sig
  type tree
  val build : ?max_leaf_size:int -> ?depth:int -> point list -> tree
  val find_k_nearest : tree -> int -> point -> point list
end

module PDFEstimator : sig
  type t
  type params = {
    bandwidth: float;
    kernel: [`Gaussian | `Epanechnikov];
    min_samples: int;
    adaptive: bool;
  }

  val default_params : params
  val create : ?params:params -> Tensor.t -> t
  val estimate_density : t -> point -> float
  val estimate_gradient : t -> point -> vector
end

module PrincipalCurve : sig
  type t
  type params = {
    max_iter: int;
    tol: float;
    min_points: int;
    smoothing: float;
  }

  val default_params : params
  val create : ?params:params -> PDFEstimator.t -> t
  val smooth_points : point list -> float -> point list
  val project : t -> point -> point
  val tangent : t -> point -> vector
  val fit : t -> Tensor.t -> t
end

module MetricTensor : sig
  type metric_type = 
    | Infomax
    | ErrorMinimization
    | Decorrelation

  type params = {
    metric_type: metric_type;
    gamma: float;
    adaptation_rate: float;
    min_eigenval: float;
  }

  val default_params : params
  val compute_metric : params -> float -> point -> matrix -> matrix
  val adapt_metric : matrix -> matrix -> params -> matrix
end

module LocalEqualization : sig
  type equalization_params = {
    neighborhood_size: int;
    min_points: int;
    smoothing: float;
  }

  val default_params : equalization_params
  val compute_local_stats : Tensor.t -> point -> equalization_params -> (point * matrix)
  val equalize_local : equalization_params -> Tensor.t -> point -> matrix -> matrix
end

module SPCA : sig
  type t
  type params = {
    n_components: int;
    max_iter: int;
    tol: float;
    metric_params: MetricTensor.params;
    equalization_params: LocalEqualization.equalization_params;
    pdf_params: PDFEstimator.params;
    curve_params: PrincipalCurve.params;
  }

  val default_params : params
  val create : ?params:params -> Tensor.t -> t
  val compute_curves : t -> Tensor.t -> PrincipalCurve.t list
  val transform : t -> point -> float list
  val inverse_transform : t -> float list -> point
end