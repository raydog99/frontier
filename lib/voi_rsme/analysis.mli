open Torch

val train_model : 
  [< `LinearRegression | `PLS of int | `NeuralNetwork ] -> 
  Tensor.t -> Tensor.t -> Model.t

val evaluate_model : Model.t -> Tensor.t -> Tensor.t -> float * float * float

val analyze_lags_and_symbols : 
  Dataset.t -> int -> int -> 
  (int * int * string * float * float * float) list