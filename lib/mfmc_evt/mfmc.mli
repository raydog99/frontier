open Torch
open Types
open Distributions

val baseline_estimator : (module Distribution with type params = 'a) -> Tensor.t -> ('a, mf_error) result
val joint_mle : (module Distribution with type params = 'a) -> Tensor.t -> Tensor.t -> ('a, mf_error) result
val moment_mf : (module Distribution with type params = 'a) -> Tensor.t -> Tensor.t -> Tensor.t -> ('a, mf_error) result
val marginal_mle : (module Distribution with type params = 'a) -> Tensor.t -> Tensor.t -> Tensor.t -> ('a, mf_error) result
val optimal_alpha : (int -> Tensor.t) -> (int -> Tensor.t) -> int -> (Tensor.t, mf_error) result
val asymptotic_variance : (module Distribution with type params = 'a) -> 'a -> int -> Tensor.t
val compare_estimators : 
  (module Distribution with type params = 'a) -> 
  'a -> int -> int -> int -> 
  (Tensor.t * Tensor.t * Tensor.t * Tensor.t)

module QoI : sig
  val exceedance_probability : distribution_params -> Tensor.t -> (Tensor.t, mf_error) result
  val extreme_quantile : distribution_params -> float -> (Tensor.t, mf_error) result
end