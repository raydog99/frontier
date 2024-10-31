open Torch

val estimate : 
  ?config:Types.estimation_config ->
  samples:Tensor.t ->
  unit ->
  Tensor.t * Types.error_bound

val estimate_multiplicative :
  samples:Tensor.t ->
  epsilon:float ->
  Tensor.t * Types.error_bound

val estimate_additive :
  samples:Tensor.t ->
  epsilon:float ->
  Tensor.t * Types.error_bound

val verify_estimate :
  estimate:Tensor.t ->
  samples:Tensor.t ->
  epsilon:float ->
  [`Multiplicative | `Additive] ->
  bool * Types.error_bound