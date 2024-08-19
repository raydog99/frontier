open Torch

let mse x y =
  Tensor.(mean (pow (x - y) (Scalar.f 2.)))

let rmse x y =
  Tensor.sqrt (mse x y)

let mae x y =
  Tensor.(mean (abs (x - y)))

let compute_metrics actual predicted =
  let mse_val = mse actual predicted |> Tensor.float_value in
  let rmse_val = rmse actual predicted |> Tensor.float_value in
  let mae_val = mae actual predicted |> Tensor.float_value in
  (mse_val, rmse_val, mae_val)