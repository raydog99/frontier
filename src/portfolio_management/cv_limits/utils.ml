open Base
open Torch
open Types

let split_data (x, y) train_ratio =
  let n = Tensor.shape x |> List.hd_exn in
  let train_size = Int.of_float (Float.of_int n *. train_ratio) in
  let train_indices = List.init train_size ~f:Fn.id in
  let test_indices = List.init (n - train_size) ~f:(fun i -> i + train_size) in
  
  let train_data = Tensor.index x (Tensor.of_int1 train_indices),
                   Tensor.index y (Tensor.of_int1 train_indices) in
  let test_data = Tensor.index x (Tensor.of_int1 test_indices),
                  Tensor.index y (Tensor.of_int1 test_indices) in
  
  { DataSplit.train_data; test_data }

let mse_loss y_pred y =
  Tensor.mse_loss y_pred y Tensor.Float

let mae_loss y_pred y =
  Tensor.abs (Tensor.sub y_pred y) |> Tensor.mean