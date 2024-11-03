open Torch

let rmse ~predicted ~target ~mask =
  let open Tensor in
  let squared_error = pow (sub predicted target) (scalar 2.) in
  let masked_error = mul squared_error mask in
  let mean_error = div (sum masked_error) (sum mask) in
  sqrt mean_error

let mae ~predicted ~target ~mask =
  let open Tensor in
  let abs_error = abs (sub predicted target) in
  let masked_error = mul abs_error mask in
  div (sum masked_error) (sum mask)

let evaluate_imputation ~imputer ~test_data =
  let imputed = Imputer.impute imputer ~time_series:test_data ~n_trajectories:50 in
  let rmse_score = rmse ~predicted:imputed ~target:test_data.data ~mask:test_data.mask in
  let mae_score = mae ~predicted:imputed ~target:test_data.data ~mask:test_data.mask in
  rmse_score, mae_score