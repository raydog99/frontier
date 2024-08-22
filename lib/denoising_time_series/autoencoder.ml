open Torch

type t = {
  encoder: Module.t;
  decoder: Module.t;
  bottleneck: Module.t;
}

let create input_channels =
  let encoder = Module.sequential [
    Module.conv1d ~input_channels ~output_channels:16 ~kernel_size:3 ~stride:1 ~padding:1 ();
    Module.relu;
    Module.max_pool1d ~kernel_size:2 ~stride:2 ();
    Module.conv1d ~input_channels:16 ~output_channels:32 ~kernel_size:3 ~stride:1 ~padding:1 ();
    Module.relu;
    Module.max_pool1d ~kernel_size:2 ~stride:2 ();
  ] in
  let bottleneck = Module.sequential [
    Module.flatten;
    Module.linear ~input_dim:512 ~output_dim:10 ();
    Module.relu;
  ] in
  let decoder = Module.sequential [
    Module.linear ~input_dim:10 ~output_dim:512 ();
    Module.relu;
    Module.reshape ~shape:[32; 16; 1];
    Module.conv_transpose1d ~input_channels:32 ~output_channels:16 ~kernel_size:3 ~stride:2 ~padding:1 ~output_padding:1 ();
    Module.relu;
    Module.conv_transpose1d ~input_channels:16 ~output_channels:input_channels ~kernel_size:3 ~stride:2 ~padding:1 ~output_padding:1 ();
  ] in
  { encoder; decoder; bottleneck }

let forward t input =
  let encoded = Module.forward t.encoder input in
  let bottlenecked = Module.forward t.bottleneck encoded in
  let decoded = Module.forward t.decoder bottlenecked in
  (bottlenecked, decoded)

let reconstruction_loss predicted target =
  Tensor.mse_loss predicted target

let train t input learning_rate =
  let bottlenecked, predicted = forward t input in
  let loss = reconstruction_loss predicted input in
  Tensor.backward loss;
  Optimizer.Adam.step (Module.parameters t.encoder) ~learning_rate;
  Optimizer.Adam.step (Module.parameters t.bottleneck) ~learning_rate;
  Optimizer.Adam.step (Module.parameters t.decoder) ~learning_rate;
  (loss, bottlenecked)