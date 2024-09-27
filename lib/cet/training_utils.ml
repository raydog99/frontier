open Torch
open Data_loader
open Cet

let train_step model optimizer price_volume earnings z_pos z_neg labels temperature =
  Optimizer.zero_grad optimizer;
  let c_t = CET.forward model price_volume earnings in
  let cpc_loss = CET.cpc_loss model c_t z_pos z_neg temperature in
  let classification_loss = CET.classification_loss model price_volume earnings labels in
  let total_loss = Tensor.(cpc_loss + classification_loss) in
  Tensor.backward total_loss;
  Optimizer.step optimizer;
  total_loss

let evaluate model data_loader =
  let total_loss = ref 0. in
  let total_correct = ref 0 in
  let total_samples = ref 0 in
  Data_loader.iter data_loader ~f:(fun (price_volume, earnings, labels) ->
    let (probabilities, predicted_classes) = CET.predict model price_volume earnings in
    let loss = Tensor.cross_entropy_loss probabilities ~targets:labels in
    total_loss := !total_loss +. Tensor.to_float0_exn loss;
    total_correct := !total_correct + Tensor.(sum (predicted_classes = labels) |> to_int0_exn);
    total_samples := !total_samples + Tensor.shape labels |> List.hd
  );
  let avg_loss = !total_loss /. float_of_int !total_samples in
  let accuracy = float_of_int !total_correct /. float_of_int !total_samples in
  (avg_loss, accuracy)

module LRScheduler = struct
  type t = {
    optimizer: Optimizer.t;
    initial_lr: float;
    decay_rate: float;
    decay_steps: int;
  }

  let create optimizer initial_lr decay_rate decay_steps =
    { optimizer; initial_lr; decay_rate; decay_steps }

  let step t epoch =
    let new_lr = t.initial_lr *. (t.decay_rate ** (float_of_int epoch /. float_of_int t.decay_steps)) in
    Optimizer.set_lr t.optimizer ~lr:new_lr
end