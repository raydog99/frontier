open Torch
open Torch.Tensor
open Torch.Nn
open Torch.Optim

type t = {
  mutable weights : Tensor.t;
  optimizer : Optimizer.t;
  scheduler : Scheduler.t option;
}

let create learning_rate =
  let weights = Tensor.zeros [5; 1] ~requires_grad:true in
  let optimizer = Adam.create [weights] ~lr:learning_rate in
  let scheduler = Some (Step_lr.create optimizer ~step_size:100 ~gamma:0.1) in
  { weights; optimizer; scheduler }

let train model factors returns num_epochs batch_size =
  for epoch = 1 to num_epochs do
    let batches = Tensor.split_n factors batch_size 0 in
    List.iteri (fun i batch_factors ->
      let start_idx = i * batch_size in
      let end_idx = min (start_idx + batch_size) (Tensor.shape returns |> List.hd) in
      let batch_returns = Tensor.narrow returns ~dim:0 ~start:start_idx ~length:(end_idx - start_idx) in
      let predictions = Tensor.mm batch_factors model.weights in
      let loss = Tensor.mse_loss predictions batch_returns in
      
      Optimizer.zero_grad model.optimizer;
      Tensor.backward loss;
      Optimizer.step model.optimizer;
    ) batches;
    
    Option.iter (fun s -> Scheduler.step s) model.scheduler;

    if epoch mod 100 = 0 then
      let current_loss = Tensor.mse_loss (Tensor.mm factors model.weights) returns |> Tensor.to_float0_exn in
      if current_loss < 1e-6 then (
        Logs.info (fun m -> m "Early stopping at epoch %d" epoch);
        raise Exit
      )
  done;
  model

let predict model factors =
  Tensor.mm factors model.weights

let calculate_r_squared model factors returns =
  let predictions = predict model factors in
  let ss_tot = Tensor.sum (Tensor.pow (Tensor.sub returns (Tensor.mean returns)) 2.) in
  let ss_res = Tensor.sum (Tensor.pow (Tensor.sub returns predictions) 2.) in
  let r_squared = 1. -. (Tensor.to_float0_exn ss_res /. Tensor.to_float0_exn ss_tot) in
  r_squared

let get_weights model =
  Tensor.to_float1 model.weights