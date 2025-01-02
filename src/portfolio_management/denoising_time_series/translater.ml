open Torch

type t = {
  network: Module.t;
}

let create input_dim output_dim hidden_dim =
  let network = Module.sequential [
    Module.linear input_dim hidden_dim;
    Module.relu;
    Module.linear hidden_dim hidden_dim;
    Module.relu;
    Module.linear hidden_dim output_dim;
  ] in
  { network }

let forward t input =
  Module.forward t.network input

let train t input target learning_rate =
  let predicted = forward t input in
  let loss = Tensor.mse_loss predicted target in
  Tensor.backward loss;
  Optimizer.Adam.step (Module.parameters t.network) ~learning_rate;
  loss