open Torch

let generate_truncated_exponential lambda n truncation_point =
  let x = Tensor.rand [n] in
  let y = Tensor.neg (Tensor.div (Tensor.log (Tensor.sub (Tensor.of_float 1.) x)) (Tensor.of_float lambda)) in
  Tensor.clamp_max y (Tensor.of_float truncation_point)

let generate_beta shape1 shape2 n =
  let x = Tensor.rand [n] in
  let y = Tensor.rand [n] in
  let x_pow = Tensor.pow x (Tensor.of_float (1. /. shape1)) in
  let y_pow = Tensor.pow y (Tensor.of_float (1. /. shape2)) in
  Tensor.div x_pow (Tensor.add x_pow y_pow)

let generate_mixture_distribution dist1 dist2 mixing_ratio n =
  let mask = Tensor.rand [n] in
  let mixture = Tensor.where_ (Tensor.lt mask (Tensor.of_float mixing_ratio)) dist1 dist2 in
  mixture