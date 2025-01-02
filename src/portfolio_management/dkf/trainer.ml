open Torch
open Deep_kalman_filter
open Logger
open Metrics

type t = {
  model: Deep_kalman_filter.t;
  optimizer: Optimizer.t;
  scheduler: Scheduler.t;
  device: Device.t;
  logger: Logger.t;
}

let create model ~learning_rate =
  let parameters = Deep_kalman_filter.parameters model in
  let optimizer = Optimizer.adam parameters ~learning_rate in
  let scheduler = Scheduler.step optimizer ~step_size:10 ~gamma:0.1 in
  let logger = Logger.create () in
  { model; optimizer; scheduler; device = Device.cuda_if_available (); logger }

let kl_divergence mean1 cov1 mean2 cov2 =
  let term1 = Tensor.logdet cov2 - Tensor.logdet cov1 in
  let term2 = Tensor.trace (Tensor.matmul (Tensor.inverse cov2) cov1) in
  let diff = Tensor.(mean2 - mean1) in
  let term3 = Tensor.matmul (Tensor.matmul (Tensor.transpose diff ~dim0:0 ~dim1:1) (Tensor.inverse cov2)) diff in
  Tensor.((term1 - float (Tensor.shape mean1 |> List.hd) + term2 + term3) / (Scalar.f 2.0))

let loss_fn predicted actual lambda =
  let predicted_state, predicted_obs = predicted in
  let actual_state, actual_obs = actual in
  let mse = Metrics.mse predicted_obs actual_obs in
  let kl = kl_divergence predicted_state (Tensor.eye (Tensor.shape predicted_state |> List.hd)) actual_state (Tensor.eye (Tensor.shape actual_state |> List.hd)) in
  Tensor.(mse + lambda * kl)

let train_step t ~x ~y ~lambda =
  Optimizer.zero_grad t.optimizer;
  let predicted = Deep_kalman_filter.forward t.model x in
  let loss = loss_fn predicted y lambda in
  Tensor.backward loss;
  Optimizer.step t.optimizer;
  Tensor.float_value loss

let train t ~train_loader ~val_loader ~num_epochs ~patience ~lambda =
  let rec train_loop epoch patience_counter best_val_loss =
    if epoch > num_epochs || patience_counter >= patience then ()
    else begin
      let total_loss = ref 0. in
      let num_batches = ref 0 in
      DataLoader.batches train_loader
      |> Seq.iter (fun batch ->
        let x = Tensor.stack (Array.map fst batch) ~dim:0 in
        let y = Tensor.stack (Array.map snd batch) ~dim:0 in
        let loss = train_step t ~x ~y ~lambda in
        total_loss := !total_loss +. loss;
        incr num_batches;
      );
      let avg_train_loss = !total_loss /. (float_of_int !num_batches) in
      Logger.log_train_loss t.logger avg_train_loss;

      let val_loss = evaluate t ~data_loader:val_loader in
      Logger.log_val_loss t.logger val_loss;

      Printf.printf "Epoch %d, Train Loss: %f, Val Loss: %f\n" epoch avg_train_loss val_loss;

      Scheduler.step t.scheduler;

      let (patience_counter, best_val_loss) =
        if val_loss < best_val_loss then (0, val_loss)
        else (patience_counter + 1, best_val_loss)
      in

      train_loop (epoch + 1) patience_counter best_val_loss
    end
  in
  train_loop 1 0 Float.infinity

let evaluate t ~data_loader =
  let total_loss = ref 0. in
  let num_batches = ref 0 in
  DataLoader.batches data_loader
  |> Seq.iter (fun batch ->
    let x = Tensor.stack (Array.map fst batch) ~dim:0 in
    let y = Tensor.stack (Array.map snd batch) ~dim:0 in
    let predicted = Deep_kalman_filter.forward t.model x in
    let loss = loss_fn predicted y (Scalar.f 0.1) in
    total_loss := !total_loss +. Tensor.float_value loss;
    incr num_batches;
  );
  !total_loss /. (float_of_int !num_batches)

let save t ~filename =
  Deep_kalman_filter.save t.model ~filename

let load t ~filename =
  Deep_kalman_filter.load t.model ~filename