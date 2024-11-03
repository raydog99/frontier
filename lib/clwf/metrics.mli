open Torch

val rmse : predicted:Tensor.t -> target:Tensor.t -> mask:mask -> Tensor.t
val mae : predicted:Tensor.t -> target:Tensor.t -> mask:mask -> Tensor.t
val evaluate_imputation : imputer:Imputer.t -> test_data:TimeSeries.t -> Tensor.t * Tensor.t