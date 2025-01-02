open Torch
open Utils

let gradient_descent f initial_x learning_rate num_iterations =
  let rec optimize x i =
    if i = 0 then x
    else
      let x' = Tensor.(x - (float learning_rate * grad f x)) in
      optimize x' (i - 1)
  in
  optimize initial_x num_iterations

let optimal_hedge g mu constraints =
  let f h = 
    let hedged_g x1 x2 = Tensor.(g x1 x2 - Hedging.dynamic h x1 x2) in
    Sensitivities.sensitivity mu constraints
  in
  gradient_descent f (Tensor.float 0.) 0.01 1000

let optimal_semi_static_hedge g mu constraints =
  let f h f_static = 
    let hedged_g x1 x2 = Tensor.(g x1 x2 - Hedging.semi_static h f_static x1 x2) in
    Sensitivities.sensitivity mu constraints
  in
  let initial_h = Tensor.float 0. in
  let initial_f _ = Tensor.float 0. in
  gradient_descent (fun x -> f (Tensor.get x 0) (fun _ -> Tensor.get x 1)) 
    (Tensor.stack [initial_h; initial_f (Tensor.float 0.)] ~dim:0) 0.01 1000