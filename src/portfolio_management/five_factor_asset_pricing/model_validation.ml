open Torch
open Factor_model

let bootstrap_validation model factors returns num_bootstraps =
  let n = Tensor.shape factors |> List.hd in
  List.init num_bootstraps (fun _ ->
    let indices = List.init n (fun _ -> Random.int n) in
    let bootstrap_factors = Tensor.index_select factors ~dim:0 ~index:(Tensor.of_int1 (Array.of_list indices)) in
    let bootstrap_returns = Tensor.index_select returns ~dim:0 ~index:(Tensor.of_int1 (Array.of_list indices)) in
    let bootstrap_model = Factor_model.create 0.01 in
    let trained_model = Factor_model.train bootstrap_model bootstrap_factors bootstrap_returns 1000 100 in
    Factor_model.calculate_r_squared trained_model bootstrap_factors bootstrap_returns
  )

let k_fold_cross_validation model factors returns k =
  let n = Tensor.shape factors |> List.hd in
  let fold_size = n / k in
  List.init k (fun i ->
    let start = i * fold_size in
    let end_ = if i = k - 1 then n else (i + 1) * fold_size in
    let train_indices = List.append (List.init start (fun x -> x)) (List.init (n - end_) (fun x -> x + end_)) in
    let test_indices = List.init (end_ - start) (fun x -> x + start) in
    let train_factors = Tensor.index_select factors ~dim:0 ~index:(Tensor.of_int1 (Array.of_list train_indices)) in
    let train_returns = Tensor.index_select returns ~dim:0 ~index:(Tensor.of_int1 (Array.of_list train_indices)) in
    let test_factors = Tensor.index_select factors ~dim:0 ~index:(Tensor.of_int1 (Array.of_list test_indices)) in
    let test_returns = Tensor.index_select returns ~dim:0 ~index:(Tensor.of_int1 (Array.of_list test_indices)) in
    let fold_model = Factor_model.create 0.01 in
    let trained_model = Factor_model.train fold_model train_factors train_returns 1000 100 in
    Factor_model.calculate_r_squared trained_model test_factors test_returns
  )