open Torch
open Base

type garch_variant = GARCH | GJR_GARCH | TGARCH

type garch_params = {
  alpha_0: float;
  alpha: float;
  beta: float;
  gamma: float option;
  variant: garch_variant;
}

type time_series = float array

type model_type = GINN | GINN_0

let garch_predict params series window_size =
  let n = Array.length series in
  let variance = Array.make n params.alpha_0 in
  for t = 1 to n - 1 do
    let prev_var = variance.(t-1) in
    let prev_return = series.(t-1) in
    let new_var = match params.variant with
    | GARCH ->
        params.alpha_0 +. params.alpha *. (prev_return ** 2.) +. params.beta *. prev_var
    | GJR_GARCH ->
        let gamma = Option.value ~default:0. params.gamma in
        let indicator = if prev_return < 0. then 1. else 0. in
        params.alpha_0 +. params.alpha *. (prev_return ** 2.) +. gamma *. indicator *. (prev_return ** 2.) +. params.beta *. prev_var
    | TGARCH ->
        let gamma = Option.value ~default:0. params.gamma in
        let abs_return = abs_float prev_return in
        params.alpha_0 +. params.alpha *. abs_return +. gamma *. (max 0. (-.prev_return)) +. params.beta *. prev_var
    in
    variance.(t) <- new_var
  done;
  Array.sub variance (n - window_size) window_size

let create_lstm_model input_size hidden_size num_layers =
  let lstm = Layer.lstm ~input_dim:input_size ~hidden_size ~num_layers () in
  let linear1 = Layer.linear ~in_features:hidden_size ~out_features:64 () in
  let linear2 = Layer.linear ~in_features:64 ~out_features:1 () in
  Staged.stage (fun x ->
    let lstm_out, _ = Layer.lstm_run lstm x false in
    let dense1 = Tensor.relu (Layer.linear linear1 (Tensor.select lstm_out (-1) (-1))) in
    Layer.linear linear2 dense1
  )

let create_ginn_model input_size hidden_size num_layers lambda model_type =
  let lstm_model = create_lstm_model input_size hidden_size num_layers in
  match model_type with
  | GINN ->
      Staged.stage (fun x garch_pred ->
        let lstm_pred = Staged.unstage lstm_model x in
        Tensor.(lambda * lstm_pred + ((Tensor.of_float1 1. - lambda) * garch_pred))
      )
  | GINN_0 ->
      Staged.stage (fun x garch_pred ->
        Staged.unstage lstm_model x
      )

let generate_random_series length =
  Array.init length (fun _ -> Random.float 2. -. 1.)

let calculate_returns prices =
  Array.init (Array.length prices - 1) (fun i ->
    log (prices.(i+1) /. prices.(i))
  )

let normalize_data data =
  let mean = Tensor.mean data in
  let std = Tensor.std data ~dim:[0] ~unbiased:true ~keepdim:true in
  Tensor.((data - mean) / std), mean, std

let prepare_data series window_size =
  let n = Array.length series in
  let x = Tensor.of_float_array2 (Array.init (n - window_size) (fun i ->
    Array.sub series i window_size
  )) in
  let y = Tensor.of_float_array (Array.sub series window_size (n - window_size)) in
  let x_norm, x_mean, x_std = normalize_data x in
  let y_norm, y_mean, y_std = normalize_data y in
  (x_norm, y_norm), (x_mean, x_std), (y_mean, y_std)

let ginn_loss lambda true_var garch_pred ginn_pred =
  let mse_true = Tensor.(mse_loss ginn_pred true_var) in
  let mse_garch = Tensor.(mse_loss ginn_pred garch_pred) in
  Tensor.(lambda * mse_true + ((Tensor.of_float0 1. - lambda) * mse_garch))

let ginn_0_loss true_var ginn_pred =
  Tensor.(mse_loss ginn_pred true_var)

let train_ginn_model model garch_params series window_size epochs learning_rate lambda model_type =
  let (x, y), (x_mean, x_std), (y_mean, y_std) = prepare_data series window_size in
  let garch_pred = 
    let garch_var = garch_predict garch_params series window_size in
    let garch_tensor = Tensor.of_float_array garch_var in
    Tensor.((garch_tensor - y_mean) / y_std)
  in
  let optimizer = Optimizer.adam (Staged.unstage model) ~learning_rate in
  
  let best_loss = ref Float.max_float in
  let patience = 50 in
  let no_improve_count = ref 0 in

  for epoch = 1 to epochs do
    let pred = Staged.unstage model x garch_pred in
    let loss = match model_type with
      | GINN -> ginn_loss lambda y garch_pred pred
      | GINN_0 -> ginn_0_loss y pred
    in
    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;

    let current_loss = Tensor.to_float0_exn loss in
    if current_loss < !best_loss then begin
      best_loss := current_loss;
      no_improve_count := 0;
    end else begin
      no_improve_count := !no_improve_count + 1;
    end;

    if epoch mod 100 = 0 then
      Stdio.printf "Epoch %d, Loss: %f\n" epoch current_loss;

    if !no_improve_count >= patience then begin
      Stdio.printf "Early stopping at epoch %d\n" epoch;
      raise Exit
    end
  done

let evaluate_model model garch_params series window_size model_type =
  let (x, y), (x_mean, x_std), (y_mean, y_std) = prepare_data series window_size in
  let garch_pred = 
    let garch_var = garch_predict garch_params series window_size in
    let garch_tensor = Tensor.of_float_array garch_var in
    Tensor.((garch_tensor - y_mean) / y_std)
  in
  let pred = Staged.unstage model x garch_pred in
  let denorm_pred = Tensor.((pred * y_std) + y_mean) in
  let denorm_y = Tensor.((y * y_std) + y_mean) in
  let mse = Tensor.(mse_loss denorm_pred denorm_y) in
  let mae = Tensor.(mean (abs (denorm_pred - denorm_y))) in
  let y_mean = Tensor.(mean denorm_y) in
  let ss_tot = Tensor.(sum (sqr (denorm_y - y_mean))) in
  let ss_res = Tensor.(sum (sqr (denorm_y - denorm_pred))) in
  let r2 = Tensor.(sub (Tensor.of_float0 1.) (div ss_res ss_tot)) in
  Tensor.to_float0_exn mse, Tensor.to_float0_exn mae, Tensor.to_float0_exn r2