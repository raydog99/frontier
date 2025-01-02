open Base
open Torch
open Types

let k_fold k data =
  let x, y = data in
  let n = Tensor.shape x |> List.hd_exn in
  let fold_size = n / k in
  List.init k ~f:(fun i ->
    let start_idx = i * fold_size in
    let end_idx = Int.min (start_idx + fold_size) n in
    let train_indices = List.concat [List.init start_idx ~f:Fn.id; List.init (n - end_idx) ~f:(fun j -> j + end_idx)] in
    let test_indices = List.init (end_idx - start_idx) ~f:(fun j -> j + start_idx) in
    let train_data = Tensor.index x (Tensor.of_int1 train_indices),
                     Tensor.index y (Tensor.of_int1 train_indices) in
    let test_data = Tensor.index x (Tensor.of_int1 test_indices),
                    Tensor.index y (Tensor.of_int1 test_indices) in
    (train_data, test_data)
  )

let leave_one_out data =
  let x, y = data in
  let n = Tensor.shape x |> List.hd_exn in
  List.init n ~f:(fun i ->
    let train_indices = List.concat [List.init i ~f:Fn.id; List.init (n - i - 1) ~f:(fun j -> j + i + 1)] in
    let test_index = [i] in
    let train_data = Tensor.index x (Tensor.of_int1 train_indices),
                     Tensor.index y (Tensor.of_int1 train_indices) in
    let test_data = Tensor.index x (Tensor.of_int1 test_index),
                    Tensor.index y (Tensor.of_int1 test_index) in
    (train_data, test_data)
  )

let stratified_k_fold k data =
  (* Simplified stratification based on y values *)
  let x, y = data in
  let n = Tensor.shape x |> List.hd_exn in
  let y_list = Tensor.to_float1d_exn y in
  let sorted_indices = List.init n ~f:Fn.id
    |> List.sort ~compare:(fun i j -> Float.compare y_list.(i) y_list.(j))
  in
  let fold_size = n / k in
  List.init k ~f:(fun i ->
    let test_indices = List.filteri sorted_indices ~f:(fun idx _ -> idx % k = i) in
    let train_indices = List.filter sorted_indices ~f:(fun idx -> not (List.mem test_indices idx ~equal:Int.equal)) in
    let train_data = Tensor.index x (Tensor.of_int1 train_indices),
                     Tensor.index y (Tensor.of_int1 train_indices) in
    let test_data = Tensor.index x (Tensor.of_int1 test_indices),
                    Tensor.index y (Tensor.of_int1 test_indices) in
    (train_data, test_data)
  )

let time_series_split n_splits data =
  let x, y = data in
  let n = Tensor.shape x |> List.hd_exn in
  let min_train_size = n / (n_splits + 1) in
  List.init n_splits ~f:(fun i ->
    let train_size = min_train_size * (i + 1) in
    let test_size = Int.min (n - train_size) (n / (n_splits + 1)) in
    let train_indices = List.init train_size ~f:Fn.id in
    let test_indices = List.init test_size ~f:(fun j -> j + train_size) in
    let train_data = Tensor.index x (Tensor.of_int1 train_indices),
                     Tensor.index y (Tensor.of_int1 train_indices) in
    let test_data = Tensor.index x (Tensor.of_int1 test_indices),
                    Tensor.index y (Tensor.of_int1 test_indices) in
    (train_data, test_data)
  )

let cross_validate cv_type data =
  match cv_type with
  | CVType.KFold k -> k_fold k data
  | CVType.LOOCV -> leave_one_out data
  | CVType.StratifiedKFold k -> stratified_k_fold k data
  | CVType.TimeSeriesSplit n -> time_series_split n data