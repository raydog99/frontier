open Torch

exception Invalid_Input of string
exception Convergence_Error of string
exception Numerical_Error of string

let validate_input tensor =
  if Tensor.shape tensor |> List.length <> 1 then
    raise (Invalid_Input "Input must be a 1D tensor")
  else if Tensor.shape tensor |> List.hd < 2 then
    raise (Invalid_Input "Input must have at least 2 elements")

let check_convergence loss tolerance =
  if Float.is_nan loss || Float.is_infinite loss then
    raise (Convergence_Error "Optimization did not converge")
  else if loss > tolerance then
    raise (Convergence_Error (Printf.sprintf "Optimization did not reach desired tolerance: %f > %f" loss tolerance))

let safe_division a b =
  if Tensor.to_float0_exn b = 0.0 then
    raise (Numerical_Error "Division by zero")
  else
    Tensor.(a / b)