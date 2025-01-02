open Base
open Torch

module Tensor_utils = struct
  let to_float1 t = Tensor.to_float1_exn t
  let of_float1 a = Tensor.of_float1 a

  let linear_regression x y =
    let x = of_float1 (Array.of_list x) in
    let y = of_float1 (Array.of_list y) in
    let n = Float.of_int (List.length x) in
    let sum_x = Tensor.sum x in
    let sum_y = Tensor.sum y in
    let sum_xy = Tensor.sum (Tensor.mul x y) in
    let sum_x2 = Tensor.sum (Tensor.mul x x) in
    let slope = (n *. sum_xy -. sum_x *. sum_y) /. (n *. sum_x2 -. sum_x *. sum_x) in
    let intercept = (sum_y -. slope *. sum_x) /. n in
    (slope, intercept)

  let moving_average t window_size =
    let n = Tensor.shape t |> Array.get 0 in
    let result = Tensor.zeros [n] in
    for i = 0 to n - 1 do
      let start = Int.max 0 (i - window_size + 1) in
      let window = Tensor.narrow t 0 start (i - start + 1) in
      let avg = Tensor.mean window in
      Tensor.set result [|i|] (Tensor.get avg [||])
    done;
    result
end

type params = {
  h : float;
  nu : float;
  alpha : float;
  m : float;
}

exception Invalid_parameter of string
exception Insufficient_data of string

let validate_params params =
  if params.h <= 0. || params.h >= 0.5 then
    raise (Invalid_parameter "Hurst parameter h must be in (0, 0.5)");
  if params.nu <= 0. then
    raise (Invalid_parameter "Volatility of volatility nu must be positive");
  if params.alpha < 0. then
    raise (Invalid_parameter "Mean reversion rate alpha must be non-negative")

let fbm h n =
  let covariance i j =
    0.5 *. (Float.abs (Float.pow (Float.of_int i) (2. *. h)) +.
            Float.abs (Float.pow (Float.of_int j) (2. *. h)) -.
            Float.abs (Float.pow (Float.of_int (i - j)) (2. *. h)))
  in
  let cov_matrix = Tensor.of_float2 (Array.make_matrix n n 0.) in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      Tensor.set cov_matrix [|i; j|] (covariance i j)
    done
  done;
  let l = Tensor.cholesky cov_matrix in
  let z = Tensor.randn [n] in
  Tensor.mm l z

let generate_rfsv params n =
  validate_params params;
  let w = fbm params.h n in
  let x = Tensor.zeros [n] in
  Tensor.set x [|0|] params.m;
  for t = 1 to n - 1 do
    let prev = Tensor.get x [|t - 1|] in
    let increment = params.nu *. (Tensor.get w [|t|] -. Tensor.get w [|t - 1|]) in
    let mean_reversion = params.alpha *. (params.m -. prev) in
    Tensor.set x [|t|] (prev +. increment +. mean_reversion)
  done;
  Tensor.exp x

let estimate_h data =
  let n = Tensor.shape data |> Array.get 0 in
  if n < 20 then
    raise (Insufficient_data "At least 20 data points are required to estimate h");
  let max_lag = Int.min 20 (n / 2) in
  let lags = Array.init max_lag ~f:(fun i -> i + 1) in
  let m_q_delta = Array.map lags ~f:(fun delta ->
    let diffs = Tensor.sub (Tensor.narrow data 0 delta (n - delta))
                           (Tensor.narrow data 0 0 (n - delta)) in
    Tensor.mean (Tensor.abs diffs)
  ) in
  let x = Array.map lags ~f:(fun lag -> Float.log (Float.of_int lag)) in
  let y = Array.map m_q_delta ~f:(fun m -> Float.log (Tensor.get m [|0|])) in
  let slope, _ = Tensor_utils.linear_regression (Array.to_list x) (Array.to_list y) in
  slope

let forecast_log_volatility data h horizon =
  let n = Tensor.shape data |> Array.get 0 in
  if n < 2 then
    raise (Insufficient_data "At least 2 data points are required for forecasting");
  let weights = Tensor.zeros [n] in
  for i = 0 to n - 1 do
    let t = Float.of_int (n - i) in
    let weight = Float.cos (Float.pi *. h) /. Float.pi *. 
                 (Float.pow horizon (h +. 0.5)) /.
                 ((t +. horizon) *. (Float.pow t (h +. 0.5)))
    in
    Tensor.set weights [|i|] weight
  done;
  let forecast = Tensor.sum (Tensor.mul data weights) in
  Tensor.get forecast [|0|]

let forecast_variance data h nu horizon =
  let log_forecast = forecast_log_volatility data h horizon in
  let c = Stdlib.Gamma.gamma (1.5 -. h) *. Stdlib.Gamma.gamma (h +. 0.5) /. Stdlib.Gamma.gamma (2. -. 2. *. h) in
  let variance_term = 2. *. c *. nu *. nu *. (Float.pow horizon (2. *. h)) in
  Float.exp (2. *. log_forecast +. variance_term)

let forecast_ar data p horizon =
  let n = Tensor.shape data |> Array.get 0 in
  if n < p + horizon then
    raise (Insufficient_data (Printf.sprintf "At least %d data points are required for AR(%d) forecasting" (p + horizon) p));
  let x = Tensor.narrow data 0 0 (n - horizon) in
  let y = Tensor.narrow data 0 horizon n in
  let coeffs = Array.create ~len:(p + 1) 0. in
  for i = 0 to p do
    let x_i = Tensor.narrow x 0 i (n - horizon - p) in
    let y_i = Tensor.narrow y 0 i (n - horizon - p) in
    let slope, intercept = Tensor_utils.linear_regression 
      (Tensor_utils.to_float1 x_i |> Array.to_list)
      (Tensor_utils.to_float1 y_i |> Array.to_list)
    in
    coeffs.(i) <- slope
  done;
  let forecast = ref 0. in
  for i = 0 to p do
    forecast := !forecast +. coeffs.(i) *. (Tensor.get data [|n - p + i - 1|])
  done;
  !forecast

let forecast_har data horizon =
  let n = Tensor.shape data |> Array.get 0 in
  if n < 22 + horizon then
    raise (Insufficient_data (Printf.sprintf "At least %d data points are required for HAR forecasting" (22 + horizon)));
  let daily = Tensor.narrow data 0 (n - 1) 1 in
  let weekly = Tensor.mean (Tensor.narrow data 0 (n - 5) 5) in
  let monthly = Tensor.mean (Tensor.narrow data 0 (n - 22) 22) in
  let x = [|Tensor.get daily [|0|]; Tensor.get weekly [|0|]; Tensor.get monthly [|0|]|] in
  let y = Tensor.get data [|n - 1 + horizon|] in
  let slope, intercept = Tensor_utils.linear_regression (Array.to_list x) [y] in
  intercept +. slope *. x.(0) +. slope *. x.(1) +. slope *. x.(2)

let prepare_forecast_data data window_size horizon =
  let n = Tensor.shape data |> Array.get 0 in
  if n < window_size + horizon then
    raise (Insufficient_data (Printf.sprintf "At least %d data points are required for the given window size and horizon" (window_size + horizon)));
  let x = Tensor.narrow data 0 0 (n - horizon) in
  let y = Tensor.narrow data 0 horizon n in
  (x, y)

let mse predictions targets =
  let diff = Tensor.sub predictions targets in
  Tensor.mean (Tensor.mul diff diff)

let calculate_p_ratio predictions targets =
  let mse_val = mse predictions targets in
  let variance = Tensor.var targets in
  Tensor.div mse_val variance

let evaluate_forecast forecast_fn data window_size horizon =
  let n = Tensor.shape data |> Array.get 0 in
  let num_forecasts = n - window_size - horizon + 1 in
  if num_forecasts <= 0 then
    raise (Insufficient_data (Printf.sprintf "Insufficient data for evaluation with window size %d and horizon %d" window_size horizon));
  let predictions = Tensor.zeros [num_forecasts] in
  let targets = Tensor.zeros [num_forecasts] in
  
  for i = 0 to num_forecasts - 1 do
    let window = Tensor.narrow data 0 i window_size in
    let forecast = forecast_fn window horizon in
    let target = Tensor.get data [|i + window_size + horizon - 1|] in
    Tensor.set predictions [|i|] forecast;
    Tensor.set targets [|i|] target
  done;
  
  calculate_p_ratio predictions targets

let rfsv_forecast params data horizon =
  let h = params.h in
  let nu = params.nu in
  forecast_variance data h nu (Float.of_int horizon)

let ar_forecast p data horizon =
  forecast_ar data p horizon

let har_forecast data horizon =
  forecast_har data horizon

let compare_forecasts data window_size horizons =
  let params = { h = estimate_h data; nu = estimate_nu data; alpha = estimate_alpha data; m = Tensor.mean data |> Tensor.get [||] } in
  
  List.iter horizons ~f:(fun horizon ->
    let rfsv_p = evaluate_forecast (rfsv_forecast params) data window_size horizon in
    let ar5_p = evaluate_forecast (ar_forecast 5) data window_size horizon in
    let ar10_p = evaluate_forecast (ar_forecast 10) data window_size horizon in
    let har_p = evaluate_forecast har_forecast data window_size horizon in
    
    Printf.printf "Horizon: %d\n" horizon;
    Printf.printf "RFSV P-ratio: %f\n" (Tensor.get rfsv_p [|0|]);
    Printf.printf "AR(5) P-ratio: %f\n" (Tensor.get ar5_p [|0|]);
    Printf.printf "AR(10) P-ratio: %f\n" (Tensor.get ar10_p [|0|]);
    Printf.printf "HAR P-ratio: %f\n\n" (Tensor.get har_p [|0|])
  )

let load_csv_data filename =
  let ic = Stdio.In_channel.create filename in
  let data = ref [] in
  try
    while true do
      let line = Stdio.In_channel.input_line_exn ic in
      let parts = String.split line ~on:',' in
      match parts with
      | _ :: value :: _ -> 
        (try
          data := Float.of_string value :: !data
        with _ -> ())
      | _ -> ()
    done;
  with
  | End_of_file ->
      Stdio.In_channel.close ic;
      let result = List.rev !data in
      if List.length result < 2 then
        raise (Insufficient_data "CSV file must contain at least 2 valid data points");
      Tensor.of_float1 (Array.of_list result)
  | e ->
      Stdio.In_channel.close ic;
      raise e

let preprocess_data data =
  let log_data = Tensor.log data in
  let returns = Tensor.sub (Tensor.narrow log_data 0 1 (Tensor.shape log_data |> Array.get 0 - 1))
                           (Tensor.narrow log_data 0 0 (Tensor.shape log_data |> Array.get 0 - 1)) in
  returns

let estimate_nu data =
  let n = Tensor.shape data |> Array.get 0 in
  if n < 20 then
    raise (Insufficient_data "At least 20 data points are required to estimate nu");
  let max_lag = Int.min 20 (n / 2) in
  let lags = Array.init max_lag ~f:(fun i -> i + 1) in
  let m_2_delta = Array.map lags ~f:(fun delta ->
    let diffs = Tensor.sub (Tensor.narrow data 0 delta (n - delta))
                           (Tensor.narrow data 0 0 (n - delta)) in
    Tensor.mean (Tensor.mul diffs diffs)
  ) in
  let x = Array.map lags ~f:(fun lag -> Float.log (Float.of_int lag)) in
  let y = Array.map m_2_delta ~f:(fun m -> Float.log (Tensor.get m [|0|])) in
  let slope, intercept = Tensor_utils.linear_regression (Array.to_list x) (Array.to_list y) in
  Float.sqrt (Float.exp intercept)

let estimate_alpha data =
  let n = Tensor.shape data |> Array.get 0 in
  if n < 2 then
    raise (Insufficient_data "At least 2 data points are required to estimate alpha");
  let acf_1 = Tensor.mean (Tensor.mul (Tensor.narrow data 0 0 (n - 1))
                                      (Tensor.narrow data 0 1 (n - 1))) in
  let variance = Tensor.var data in
  -. Float.log (Tensor.get acf_1 [|0|] /. Tensor.get variance [|0|])

let estimate_params data =
  {
    h = estimate_h data;
    nu = estimate_nu data;
    alpha = estimate_alpha data;
    m = Tensor.mean data |> Tensor.get [||];
  }

let tensor_to_list t =
  Tensor_utils.to_float1 t |> Array.to_list

let list_to_tensor l =
  Tensor_utils.of_float1 (Array.of_list l)

let rolling_volatility data window_size =
  let returns = preprocess_data data in
  let squared_returns = Tensor.mul returns returns in
  Tensor_utils.moving_average squared_returns window_size
  |> Tensor.sqrt

let calculate_var data confidence_level horizon =
  let sorted_data = Tensor.sort data ~descending:false in
  let n = Tensor.shape sorted_data |> Array.get 0 in
  let index = Int.of_float (Float.of_int n *. (1. -. confidence_level)) in
  Tensor.get sorted_data [|index|] *. Float.sqrt (Float.of_int horizon)

let calculate_es data confidence_level horizon =
  let sorted_data = Tensor.sort data ~descending:false in
  let n = Tensor.shape sorted_data |> Array.get 0 in
  let cutoff_index = Int.of_float (Float.of_int n *. (1. -. confidence_level)) in
  let tail = Tensor.narrow sorted_data 0 0 cutoff_index in
  Tensor.mean tail |> Tensor.get [||] |> ( *. ) (Float.sqrt (Float.of_int horizon))

let optimize_params data =
  let n = Tensor.shape data |> Array.get 0 in
  let initial_params = estimate_params data in
  let learning_rate = 0.01 in
  let num_iterations = 1000 in
  
  let loss_fn params =
    let simulated_data = generate_rfsv params n in
    mse simulated_data data
  in
  
  let rec optimize params iter =
    if iter = 0 then params
    else
      let loss = loss_fn params in
      let grad_h = (loss_fn {params with h = params.h +. 1e-6} -. loss) /. 1e-6 in
      let grad_nu = (loss_fn {params with nu = params.nu +. 1e-6} -. loss) /. 1e-6 in
      let grad_alpha = (loss_fn {params with alpha = params.alpha +. 1e-6} -. loss) /. 1e-6 in
      let grad_m = (loss_fn {params with m = params.m +. 1e-6} -. loss) /. 1e-6 in
      let new_params = {
        h = Float.max 0.01 (Float.min 0.49 (params.h -. learning_rate *. grad_h));
        nu = Float.max 0. (params.nu -. learning_rate *. grad_nu);
        alpha = Float.max 0. (params.alpha -. learning_rate *. grad_alpha);
        m = params.m -. learning_rate *. grad_m;
      } in
      optimize new_params (iter - 1)
  in
  
  optimize initial_params num_iterations

let batch_forecast forecast_fn data window_size horizon batch_size =
  let n = Tensor.shape data |> Array.get 0 in
  let num_forecasts = n - window_size - horizon + 1 in
  let num_batches = (num_forecasts + batch_size - 1) / batch_size in
  let predictions = Tensor.zeros [num_forecasts] in
  
  for i = 0 to num_batches - 1 do
    let start = i * batch_size in
    let end_ = Int.min (start + batch_size) num_forecasts in
    let batch_windows = Tensor.stack (List.init (end_ - start) ~f:(fun j ->
      Tensor.narrow data 0 (start + j) window_size
    )) in
    let batch_predictions = Tensor.stack (List.init (end_ - start) ~f:(fun _ ->
      forecast_fn batch_windows horizon
    )) in
    Tensor.narrow_copy_ predictions start (end_ - start) batch_predictions
  done;
  
  predictions