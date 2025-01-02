open Torch
open Utils.Distributions

let black_scholes s0 sigma t =
  let z1 = normal (Tensor.float 0.) (Tensor.float 1.) in
  let z2 = normal (Tensor.float 0.) (Tensor.float 1.) in
  let x1 = Tensor.(exp (float (-0.5) * (sigma ** float 2.) * t + sigma * (sqrt t) * z1)) in
  let x2 = Tensor.(exp (float (-0.5) * (sigma ** float 2.) * (float 2. * t) + sigma * (sqrt t) * (z1 + z2))) in
  (x1, x2)

let bachelier s0 sigma t =
  let z1 = normal (Tensor.float 0.) (Tensor.float 1.) in
  let z2 = normal (Tensor.float 0.) (Tensor.float 1.) in
  let x1 = Tensor.(s0 + sigma * (sqrt t) * z1) in
  let x2 = Tensor.(s0 + sigma * (sqrt t) * (z1 + z2)) in
  (x1, x2)