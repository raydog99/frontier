open Torch
open Elicitable_functional

let mean b =
  Elicitable_functional.create
    (Printf.sprintf "B_Homogeneous_Mean_%f" b)
    (Scoring_function.b_homogeneous_mean b)
    1

let var b alpha =
  let score z y =
    let g = 
      match b with
      | b when b > 0. -> 
          fun x -> 
            if x > 0. then Tensor.pow x (Tensor.of_float b)
            else Tensor.neg (Tensor.pow (Tensor.neg x) (Tensor.of_float b))
      | b when b < 0. -> fun x -> Tensor.neg (Tensor.pow x (Tensor.of_float b))
      | _ -> fun x -> Tensor.log x
    in
    let indicator = Tensor.lt y z in
    let term1 = Tensor.where_ indicator (g y) (g z) in
    let term2 = Tensor.mul (Tensor.of_float alpha) (Tensor.sub z y) in
    Tensor.add term1 term2
  in
  Elicitable_functional.create
    (Printf.sprintf "B_Homogeneous_VaR_%f_%f" b alpha)
    score
    1

let es b alpha =
  let score z1 z2 y =
    let g1 = 
      match b with
      | b when b > 0. -> 
          fun x -> (Tensor.pow x (Tensor.of_float b)) 
      | _ -> fun x -> Tensor.neg (Tensor.pow x (Tensor.of_float b))
    in
    let g2 = fun x -> Tensor.pow x (Tensor.of_float b) in
    let indicator = Tensor.lt y z1 in
    let term1 = Tensor.where_ indicator (g1 y) (g1 z1) in
    let term2 = Tensor.mul (Tensor.of_float (1. -. alpha)) (Tensor.sub (g1 z1) (Tensor.mul z1 (g2 z2))) in
    let term3 = Tensor.mul (Tensor.of_float (1. -. alpha)) (g2 z2) in
    Tensor.add (Tensor.add term1 term2) term3
  in
  Elicitable_functional.create
    (Printf.sprintf "B_Homogeneous_ES_%f_%f" b alpha)
    score
    2