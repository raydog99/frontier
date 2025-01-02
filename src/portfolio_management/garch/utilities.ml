open Torch

let rolling_window f tensor window_size =
  let n = Tensor.shape tensor |> List.hd in
  let result = Tensor.zeros [n - window_size + 1] in
  for i = 0 to n - window_size do
    let window = Tensor.slice tensor ~dim:0 ~start:i ~end:(i + window_size) in
    Tensor.set result [i] (f window);
  done;
  result

let moving_average tensor window_size =
  rolling_window Tensor.mean tensor window_size

let exponential_moving_average tensor alpha =
  let n = Tensor.shape tensor |> List.hd in
  let result = Tensor.zeros_like tensor in
  Tensor.set result [0] (Tensor.get tensor [0]);
  for i = 1 to n - 1 do
    let prev = Tensor.get result [i-1] in
    let curr = Tensor.get tensor [i] in
    Tensor.set result [i] Tensor.(alpha * curr + (Scalar.f (1.0 -. alpha)) * prev);
  done;
  result

let scale_data tensor =
  let mean = Tensor.mean tensor in
  let std = Tensor.std tensor ~unbiased:true in
  Tensor.((tensor - mean) / std)

let train_test_split tensor train_ratio =
  let n = Tensor.shape tensor |> List.hd in
  let train_size = int_of_float (float_of_int n *. train_ratio) in
  let train = Tensor.slice tensor ~dim:0 ~start:0 ~end:(Some train_size) in
  let test = Tensor.slice tensor ~dim:0 ~start:train_size ~end:None in
  (train, test)