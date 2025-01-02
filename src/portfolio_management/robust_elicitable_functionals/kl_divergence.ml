open Torch

let calculate q p =
  let q_log = Tensor.log q in
  let p_log = Tensor.log p in
  let kl = Tensor.mean (Tensor.mul q (Tensor.sub q_log p_log)) in
  Tensor.to_float0_exn kl

let constraint_binding q p epsilon =
  abs_float (calculate q p -. epsilon) < 1e-6