open Torch

val analyze_ood_prediction :
  omega:Tensor.t ->
  omega_hat:Tensor.t ->
  sigma:Tensor.t ->
  psi:Tensor.t ->
  epsilon:float ->
  float * float
(** Analyze out-of-distribution prediction *)

val compute_risk_bound :
  omega:Tensor.t ->
  omega_hat:Tensor.t ->
  sigma:Tensor.t ->
  gamma:float ->
  epsilon:float ->
  float
(** Compute risk bound for out-of-distribution prediction *)

val verify_ood_conditions :
  omega:Tensor.t ->
  psi:Tensor.t ->
  epsilon:float ->
  bool
(** Verify conditions for out-of-distribution analysis *)