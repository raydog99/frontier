open Torch

type error_stats = {
  mse: float;
  mae: float;
  max_error: float;
  bias: float;
  std_dev: float;
}

let compute_error_stats true_values estimated_values =
  let n = Tensor.shape true_values |> List.hd |> float_of_int in
  let diff = Tensor.(true_values - estimated_values) in
  let mse = Tensor.(mean (pow diff (scalar 2.))) |> Tensor.to_float0_exn in
  let mae = Tensor.(mean (abs diff)) |> Tensor.to_float0_exn in
  let max_error = Tensor.(max (abs diff)) |> Tensor.to_float0_exn in
  let bias = Tensor.mean diff |> Tensor.to_float0_exn in
  let var = Tensor.(mean (pow (diff - (f bias)) (scalar 2.))) |> Tensor.to_float0_exn in
  let std_dev = sqrt var in
  { mse; mae; max_error; bias; std_dev }

let confidence_interval estimated_value error_stats confidence_level =
  let z = match confidence_level with
    | 0.90 -> 1.645
    | 0.95 -> 1.960
    | 0.99 -> 2.576
    | _ -> invalid_arg "Unsupported confidence level" in
  let margin = z *. error_stats.std_dev in
  (estimated_value -. margin, estimated_value +. margin)

let relative_error true_value estimated_value =
  abs_float (true_value -. estimated_value) /. abs_float true_value