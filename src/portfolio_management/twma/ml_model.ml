open Torch

type t = {
  model: Torch.nn;
  optimizer: Optimizer.t;
  loss_fn: Tensor.t -> Tensor.t -> Tensor.t;
}

let create input_size hidden_size output_size learning_rate =
  let model = Torch.nn [
    Linear (input_size, hidden_size);
    ReLU;
    Linear (hidden_size, output_size);
  ] in
  let optimizer = Optimizer.adam (Torch.nn_parameters model) ~learning_rate in
  let loss_fn = Tensor.mse_loss in
  { model; optimizer; loss_fn }

let train t inputs targets num_epochs =
  for _ = 1 to num_epochs do
    let predicted = Torch.nn_forward t.model inputs in
    let loss = t.loss_fn predicted targets in
    Optimizer.zero_grad t.optimizer;
    Tensor.backward loss;
    Optimizer.step t.optimizer;
    Printf.printf "Loss: %f\n" (Tensor.to_float0_exn loss)
  done

let predict t inputs =
  Torch.no_grad (fun () -> Torch.nn_forward t.model inputs)

let save t filename =
  Serialize.save_to_file t.model filename

let load t filename =
  Serialize.load_from_file filename