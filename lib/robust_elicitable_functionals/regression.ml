open Torch
open Elicitable_functional

type t = {
  functional : Elicitable_functional.t;
  coefficients : Tensor.t;
}

let create functional num_features =
  { 
    functional;
    coefficients = Tensor.randn [num_features + 1; functional.dimension];
  }

let predict regression x =
  let x_with_bias = Tensor.cat [Tensor.ones [Tensor.shape x |> List.hd; 1]; x] 1 in
  Tensor.matmul x_with_bias regression.coefficients

let loss regression x y =
  let y_pred = predict regression x in
  Tensor.mean (regression.functional.scoring_function y_pred y)

let fit regression x y learning_rate num_epochs =
  let optimizer = Optimizer.adam [regression.coefficients] ~learning_rate in
  for epoch = 1 to num_epochs do
    Optimizer.zero_grad optimizer;
    let loss = loss regression x y in
    Tensor.backward loss;
    Optimizer.step optimizer;
    if epoch mod 100 = 0 then
      Printf.printf "Epoch %d: Loss = %f\n" epoch (Tensor.to_float0_exn loss)
  done