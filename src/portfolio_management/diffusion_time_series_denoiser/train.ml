open Torch
open Score_network

let train_step optimizer score_net sde x =
  Optimizer.zero_grad optimizer;
  let t = Tensor.uniform ~low:0. ~high:1. [1] in
  let loss = Score_network.loss score_net sde x (Tensor.to_float0_exn t) in
  Tensor.backward loss;
  Optimizer.step optimizer;
  loss

let train score_net sde data_loader num_epochs learning_rate =
  let optimizer = Optimizer.adam (Nn.Module.parameters score_net.Score_network.model) ~lr:learning_rate in
  for epoch = 1 to num_epochs do
    let total_loss = ref 0. in
    Dataset.iter data_loader ~f:(fun batch ->
      let loss = train_step optimizer score_net sde batch in
      total_loss := !total_loss +. Tensor.to_float0_exn loss
    );
    Printf.printf "Epoch %d, Loss: %f\n" epoch (!total_loss /. float (Dataset.length data_loader));
  done