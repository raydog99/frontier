open Torch
open Types
open Distributions

val estimate_qoi : 
    (module Distribution with type params = 'a) -> 
    ('a -> 'b) -> 'a -> 'b

val qoi_asymptotic_variance : 
    (module Distribution with type params = 'a) -> 
    ('a -> Tensor.t) -> 'a -> int -> Tensor.t

val fit_regression : Tensor.t -> Tensor.t -> float * float
val predict_regression : float -> float -> Tensor.t -> Tensor.t
val regression_mf_estimate : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

val importance_sampling : 
    (module Distribution with type params = 'a) ->
    'a -> 'a -> int -> Tensor.t * Tensor.t

val analyze_bivariate_gaussian : unit -> unit
val analyze_bivariate_gumbel : unit -> unit
val analyze_binary_outcomes : unit -> unit