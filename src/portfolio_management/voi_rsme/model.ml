open Torch

type t =
  | LinearRegression of { weights: Tensor.t; bias: Tensor.t }
  | PLS of { weights: Tensor.t; bias: Tensor.t; components: int }
  | NeuralNetwork of { network: Tensor.t Sequential.t }

let predict model input =
  match model with
  | LinearRegression { weights; bias } ->
      Tensor.(mm input weights + bias)
  | PLS { weights; bias; _ } ->
      Tensor.(mm input weights + bias)
  | NeuralNetwork { network } ->
      Sequential.forward network input

let train_linear_regression predictors response =
  let open Tensor in
  try
    (* Compute weights using normal equations *)
    let xt_x = mm (transpose predictors 0 1) predictors in
    let xt_y = mm (transpose predictors 0 1) response in
    let weights = Linalg.solve xt_x xt_y in
    let bias = mean (sub response (mm predictors weights)) ~dim:[0] in
    LinearRegression { weights; bias }
  with _ ->
    failwith "Failed to train linear regression model"

let train_pls predictors response components =
  let open Tensor in
  try
    (* SIMPLS algorithm *)
    let x_mean = mean predictors ~dim:[0] ~keepdim:true in
    let y_mean = mean response ~dim:[0] ~keepdim:true in
    let x_centered = sub predictors x_mean in
    let y_centered = sub response y_mean in
    
    let rec pls_iteration x y t w p =
      if t = components then (w, p)
      else
        let s = mm (transpose x 0 1) y in
        let u = div s (norm s) in
        let t' = mm x u in
        let c = div (mm (transpose y 0 1) t') (dot t' t') in
        let w' = cat [w; transpose u 0 1] ~dim:0 in
        let p' = cat [p; transpose (mm (transpose x 0 1) t') 0 1] ~dim:0 in
        let x' = sub x (mm t' (transpose (slice p' [Some t; None]) 0 1)) in
        let y' = sub y (mul t' c) in
        pls_iteration x' y' (t + 1) w' p'
    in
    
    let w, p = pls_iteration x_centered y_centered 0 (zeros [0; Tensor.size predictors 1]) (zeros [0; Tensor.size predictors 1]) in
    let weights = mm (mm p (Linalg.pinv (mm (transpose w 0 1) p))) (transpose w 0 1) in
    let bias = sub y_mean (mm x_mean weights) in
    PLS { weights; bias; components }
  with _ ->
    failwith "Failed to train PLS model"

let train_neural_network predictors response =
  try
    let input_size = Tensor.size predictors 1 in
    let hidden_size = 3 in
    (* Create a simple feedforward neural network *)
    let network = Sequential.of_list [
      Linear.create input_size hidden_size;
      Activation.sigmoid;
      Linear.create hidden_size 1
    ] in
    let optimizer = Optimizer.adam (Sequential.parameters network) ~lr:0.01 in
    (* Train for 30 epochs *)
    for _ = 1 to 30 do
      let predicted = Sequential.forward network predictors in
      let loss = Tensor.mse_loss predicted response in
      Optimizer.zero_grad optimizer;
      Tensor.backward loss;
      Optimizer.step optimizer
    done;
    NeuralNetwork { network }
  with _ ->
    failwith "Failed to train neural network model"