open Torch

let check_square_matrix m =
  let rows = Tensor.size m 0 in
  let cols = Tensor.size m 1 in
  if rows != cols then
    failwith "Matrix must be square"

let check_positive_scalar s name =
  if Tensor.item s <= 0. then
    failwith (Printf.sprintf "%s must be positive" name)

let check_nonnegative_tensor t name =
  if Tensor.lt t (Tensor.zeros_like t) |> Tensor.any |> Tensor.item then
    failwith (Printf.sprintf "%s must be non-negative" name)

let check_valid_time t name =
  if t < 0. then
    failwith (Printf.sprintf "%s must be non-negative" name)

let check_valid_state ctmc state =
  if state < 0 || state >= ctmc.CTMC.state_space then
    failwith (Printf.sprintf "Invalid state: %d" state)