open Torch

module RegularizationFamily : sig
  type t = {
    g_lambda: float -> float;
    qualification: float;
    lambda_bound: float;
  }

  val create : (float -> float) -> float -> float -> t
  val ridge : float -> t
  val pca : float -> t
  val gradient_flow : float -> t
end

module SpectralAlgorithm : sig
  type t = {
    regularization: RegularizationFamily.t;
    truncation: int;
    lambda: float;
  }

  module SampleCovariance : sig
    type t = {
      eigenvalues: Tensor.t;
      eigenvectors: Tensor.t;
      points: Tensor.t;
      heat_kernel: HeatKernel.t;
    }

    val create : Tensor.t -> HeatKernel.t -> t
    val apply : t -> Tensor.t -> Tensor.t
  end

  val create : RegularizationFamily.t -> int -> float -> t
  val estimate : t -> Dataset.labeled -> Dataset.unlabeled -> HeatKernel.t -> Tensor.t
end

module DiffusionSpectralAlgorithm : sig
  type t = {
    config: Config.t;
    heat_kernel: HeatKernel.t option;
    eigensystem: SpectralAlgorithm.SampleCovariance.t option;
    regularization: RegularizationFamily.t;
  }

  val create : Config.t -> RegularizationFamily.t -> t
  val fit : t -> Dataset.labeled -> Dataset.unlabeled -> Tensor.t
end

module RegressionPipeline : sig
  val run : Config.t -> Dataset.labeled -> Dataset.unlabeled -> Tensor.t
  val run_with_power_space : 
    Config.t -> Dataset.labeled -> Dataset.unlabeled -> float -> Tensor.t
end