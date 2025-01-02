open Torch
open Types

let tensor_to_float_array (t: Tensor.t) : float array =
  Tensor.to_float1 t |> Array.of_list

let float_array_to_tensor (arr: float array) : Tensor.t =
  Tensor.of_float1 (Array.to_list arr)

let eigen_decomposition (matrix: Tensor.t) : eigen_decomposition =
  let e, v = Tensor.symeig matrix ~eigenvectors:true in
  { eigenvalues = e; eigenvectors = v }

let sort_eigenpairs (eig: eigen_decomposition) : eigen_decomposition =
  let sorted_indices = Tensor.argsort eig.eigenvalues ~descending:true in
  let sorted_eigenvalues = Tensor.index_select eig.eigenvalues 0 sorted_indices in
  let sorted_eigenvectors = Tensor.index_select eig.eigenvectors 1 sorted_indices in
  { eigenvalues = sorted_eigenvalues; eigenvectors = sorted_eigenvectors }

let validate_portfolio (portfolio: portfolio) : unit =
  let n = Array.length portfolio.assets in
  let weights_sum = Tensor.sum portfolio.weights |> Tensor.item in
  if abs_float (weights_sum -. 1.0) > 1e-6 then
    failwith "Portfolio weights do not sum to 1";
  if Tensor.shape portfolio.weights <> [n] then
    failwith "Mismatch between number of assets and weights";
  if Tensor.shape portfolio.expected_returns <> [n] then
    failwith "Mismatch between number of assets and expected returns"

let log_message (msg: string) : unit =
  Printf.printf "[%s] %s\n" (Sys.time () |> int_of_float |> string_of_int) msg