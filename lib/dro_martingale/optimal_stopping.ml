open Torch
open Utils
open Sensitivities

let value_function l1 l2 =
  let open Tensor in
  fun x1 x2 ->
    max (l1 x1) (l2 x2)

let optimal_stopping_time l1 l2 x1 x2 =
  let v = value_function l1 l2 x1 x2 in
  Tensor.(v = l1 x1)

let sensitivity mu l1 l2 constraints =
  let v = value_function l1 l2 in
  Sensitivities.sensitivity mu constraints

let adapted_sensitivity mu l1 l2 constraints =
  let v = value_function l1 l2 in
  Sensitivities.adapted_sensitivity mu constraints

let optimal_stopping_sensitivity_formula mu l1 l2 =
  let x1 = Tensor.select mu 0 in
  let x2 = Tensor.select mu 1 in
  let dx1_l1 = Tensor.grad l1 x1 in
  let dx2_l2 = Tensor.grad l2 x2 in
  let indicator = optimal_stopping_time l1 l2 x1 x2 in
  let dx1 = Tensor.(indicator * dx1_l1 + (float 1. - indicator) * dx2_l2) in
  let dx2 = Tensor.(float 1. - dx1) in
  Tensor.(sqrt (mean dx1 ** float 2. + mean dx2 ** float 2.))