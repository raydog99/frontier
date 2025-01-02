open Torch

let rmse predictions response =
  try
    let mse = Tensor.mse_loss predictions response in
    Tensor.sqrt mse |> Tensor.to_float0_exn
  with _ ->
    failwith "Failed to calculate RMSE"

let correlation predictions response =
  try
    let open Tensor in
    let p_mean = mean predictions ~dim:[0] ~keepdim:true in
    let r_mean = mean response ~dim:[0] ~keepdim:true in
    let p_centered = sub predictions p_mean in
    let r_centered = sub response r_mean in
    let numerator = sum (mul p_centered r_centered) in
    let denominator = sqrt (mul (sum (pow p_centered 2.)) (sum (pow r_centered 2.))) in
    div numerator denominator |> to_float0_exn
  with _ ->
    failwith "Failed to calculate correlation"

let mean_rate_of_return predictions response =
  try
    let open Tensor in
    let sign_match = mul (sign predictions) (sign response) in
    let returns = mul sign_match (abs response) in
    let mean_log_return = mean returns |> to_float0_exn in
    exp mean_log_return -. 1.
  with _ ->
    failwith "Failed to calculate mean rate of return"