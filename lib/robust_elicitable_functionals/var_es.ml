open Torch
open Scoring_function

let create alpha =
  let score z y =
    let z1 = Tensor.select z 1 0 in
    let z2 = Tensor.select z 1 1 in
    let var_part = Scoring_function.var_score alpha z1 y in
    let es_part = Tensor.relu (Tensor.sub y z1) in
    Tensor.add var_part (Tensor.mul (Tensor.of_float (1. /. (1. -. alpha))) es_part)
  in
  Elicitable_functional.create
    (Printf.sprintf "VaR_ES_%f" alpha)
    score
    2

let evaluate = Elicitable_functional.evaluate