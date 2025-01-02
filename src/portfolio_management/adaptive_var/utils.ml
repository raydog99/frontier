open Torch
open Types
open Config

let indicator_function x =
  Tensor.(if_ (x <= (f 0.)) (f 1.) (f 0.))

let h config l = 
  if l < 0 then raise (Error (InvalidParameter "l must be non-negative"))
  else config.h0 /. (float_of_int (config.m ** l))

let saturation_factor framework n =
  if n <= 0 then raise (Error (InvalidParameter "n must be positive"))
  else
    match framework with
    | Lipschitz_gradient -> (float_of_int n) ** (-1. /. Config.default.kappa)
    | Holder_density | Subexponential_deviation -> 
        sqrt (1. +. log (float_of_int n))

let step_size config n =
  if n <= 0 then raise (Error (InvalidParameter "n must be positive"))
  else config.gamma *. (float_of_int n) ** (-config.kappa)

let optimal_iterations config eps framework l =
  let k = Framework.optimal_complexity framework eps in
  let h_l = h config l in
  int_of_float (k *. (Framework.epsilon framework h_l) ** (1. +. config.theta))