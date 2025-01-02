open Types
open Model

val plug_in : Model.t -> DataType.t -> epochs:int -> EvaluationResult.t
val k_fold_cv : int -> Model.t -> DataType.t -> epochs:int -> EvaluationResult.t
val loocv : Model.t -> DataType.t -> epochs:int -> EvaluationResult.t
val stratified_k_fold_cv : int -> Model.t -> DataType.t -> epochs:int -> EvaluationResult.t
val time_series_cv : int -> Model.t -> DataType.t -> epochs:int -> EvaluationResult.t