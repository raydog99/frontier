open Torch

module DataType = struct
  type t = Tensor.t * Tensor.t
end

module ModelType = struct
  type t = {
    forward: Tensor.t -> Tensor.t;
    parameters: unit -> Tensor.t list;
  }
end

module LossType = struct
  type t = Tensor.t -> Tensor.t -> Tensor.t
end

module DataSplit = struct
  type t = {
    train_data: DataType.t;
    test_data: DataType.t;
  }
end

module EvaluationResult = struct
  type t = {
    point_estimate: float;
    confidence_interval: float * float;
    standard_error: float;
  }
end

module ExperimentResult = struct
  type t = {
    model_name: string;
    plug_in: EvaluationResult.t;
    cv: EvaluationResult.t;
    loocv: EvaluationResult.t;
    loo_stability: float;
    convergence_rate: float;
  }
end

module CVType = struct
  type t =
    | KFold of int
    | LOOCV
    | StratifiedKFold of int
    | TimeSeriesSplit of int
end