open Base
open Torch

type order = Zero | First | Second

let kernel_zero x x' =
  let min_x = Tensor.min x x' ~dim:1 ~keepdim:true in
  let count = Tensor.sum (Tensor.ge x min_x) ~dim:1 in
  Tensor.pow (Tensor.of_float 2.) count

let kernel_first x x' =
  let k0 = kernel_zero x x' in
  let diff = Tensor.sub x x' in
  let prod = Tensor.prod (Tensor.add diff (Tensor.of_float 1.)) ~dim:1 in
  Tensor.mul k0 prod

let kernel_second x x' =
  let k1 = kernel_first x x' in
  let diff = Tensor.sub x x' in
  let prod = Tensor.prod (Tensor.add diff (Tensor.of_float 2.)) ~dim:1 in
  Tensor.mul k1 prod

let kernel order x x' =
  match order with
  | Zero -> kernel_zero x x'
  | First -> kernel_first x x'
  | Second -> kernel_second x x'

let kernel_function order x =
  let kernel_fn i j =
    let xi = Tensor.select x 0 i in
    let xj = Tensor.select x 0 j in
    Tensor.to_float0_exn (kernel order xi xj)
  in
  kernel_fn