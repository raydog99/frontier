open Torch
open Type

let predict_samples x_new samples alpha =
  let n_samples = List.length samples in
  let n_new = size x_new 0 in
  
  (* Get predictions for each sample *)
  let predictions = List.map (fun s ->
    matmul x_new s.theta_star
  ) samples in
  
  (* Compute mean prediction *)
  let mean = List.fold_left (fun acc p -> add acc p) (zeros [n_new]) predictions in
  let mean = scalar_mul (1. /. float_of_int n_samples) mean in
  
  (* Compute standard deviation *)
  let sq_diff = List.fold_left (fun acc p ->
    let diff = sub p mean in
    add acc (mul diff diff)
  ) (zeros [n_new]) predictions in
  let std = sqrt (scalar_mul (1. /. float_of_int (n_samples - 1)) sq_diff) in
  
  (* Compute credible intervals *)
  let sorted_preds = Array.make n_new [] in
  List.iter (fun p ->
    for i = 0 to n_new - 1 do
      sorted_preds.(i) <- (get p i) :: sorted_preds.(i)
    done
  ) predictions;
  
  let lower = zeros [n_new] in
  let upper = zeros [n_new] in
  
  for i = 0 to n_new - 1 do
    let sorted = List.sort compare sorted_preds.(i) |> Array.of_list in
    let lower_idx = int_of_float (float_of_int n_samples *. alpha /. 2.) in
    let upper_idx = int_of_float (float_of_int n_samples *. (1. -. alpha /. 2.)) in
    tensor_set lower [i] sorted.(lower_idx);
    tensor_set upper [i] sorted.(upper_idx)
  done;
  
  {mean; lower; upper; std}

let compute_metrics y_true y_pred =
  let diff = sub y_true y_pred in
  let mse = mean (mul diff diff) |> float_of_elt in
  let mae = mean (abs diff) |> float_of_elt in
  
  (* R² calculation *)
  let y_mean = mean y_true in
  let total_ss = sum (pow (sub y_true y_mean) 2.) |> float_of_elt in
  let residual_ss = sum (pow diff 2.) |> float_of_elt in
  let r2 = 1. -. (residual_ss /. total_ss) in
  
  mse, mae, r2