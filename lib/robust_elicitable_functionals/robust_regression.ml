open Torch
open Regression

type t = {
  base_regression : Regression.t;
  epsilon : float;
}

let create base_regression epsilon =
  { base_regression; epsilon }

let optimize_eta regression x y z =
  let rec binary_search low high =
    if high -. low < 1e-6 then
      (low +. high) /. 2.
    else
      let mid = (low +. high) /. 2. in
      let y_pred = Regression.predict regression.base_regression x in
      let scores = regression.base_regression.functional.scoring_function y_pred y in
      let q = Tensor.softmax (Tensor.mul scores (Tensor.of_float mid)) in
      if Kl_divergence.constraint_binding q y regression.epsilon then
        mid
      else if Kl_divergence.calculate q y > regression.epsilon then
        binary_search low mid
      else
        binary_search mid high
  in
  binary_search 0. 100.

let loss regression x y =
  let eta_star = optimize_eta regression x y in
  let y_pred = Regression.predict regression.base_regression x in
  let scores = regression.base_regression.functional.scoring_function y_pred y in
  let q_star = Tensor.softmax (Tensor.mul scores (Tensor.of_float eta_star)) in
  Tensor.mean (Tensor.mul q_star scores)

let fit regression x y learning_rate num_epochs =
  let optimizer = Optimizer.adam [regression.base_regression.coefficients] ~learning_rate in
  for epoch = 1 to num_epochs do
    Optimizer.zero_grad optimizer;
    let loss = loss regression x y in
    Tensor.backward loss;
    Optimizer.step optimizer;
    if epoch mod 100 = 0 then
      Printf.printf "Epoch %d: Loss = %f\n" epoch (Tensor.to_float0_exn loss)
  done