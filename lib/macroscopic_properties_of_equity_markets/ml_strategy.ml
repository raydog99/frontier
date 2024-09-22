open Torch

let model = Nn.sequential
  [
    Nn.linear ~in_dim:10 ~out_dim:64;
    Nn.relu;
    Nn.linear ~in_dim:64 ~out_dim:1;
  ]

let optimizer = Optimizer.adam model.parameters ~learning_rate:0.01

let train_model features targets =
  for _ = 1 to 1000 do
    let predicted = Nn.Module.forward model features in
    let loss = Tensor.mse_loss predicted targets in
    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;
  done

let predict features =
  Nn.Module.forward model features

let calculate_ml_portfolio historical_data current_weights =
  let features = Tensor.narrow historical_data ~dim:0 ~start:(-10) ~length:10 in
  let predictions = predict features in
  let scaled_predictions = Tensor.softmax predictions ~dim:0 ~dtype:(T Float) in
  Tensor.mul current_weights scaled_predictions