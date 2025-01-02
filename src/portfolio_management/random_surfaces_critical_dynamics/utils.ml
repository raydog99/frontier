open Torch
open Logging

let gaussian_noise () =
  Tensor.randn [1]

let geometric_brownian_motion initial_value drift volatility dt =
  let noise = gaussian_noise () in
  initial_value *. (1.0 +. drift *. dt +. volatility *. Tensor.get noise [0] *. sqrt dt)

let linear_regression x y =
  let n = Tensor.shape x |> List.hd |> float_of_int in
  let sum_x = Tensor.sum x in
  let sum_y = Tensor.sum y in
  let sum_xy = Tensor.sum (Tensor.mul x y) in
  let sum_x_squared = Tensor.sum (Tensor.mul x x) in
  
  let slope = (n *. sum_xy -. sum_x *. sum_y) /. (n *. sum_x_squared -. sum_x *. sum_x) in
  let intercept = (sum_y -. slope *. sum_x) /. n in
  
  (slope, intercept)

let log_binning data num_bins =
  let sorted_data = Tensor.sort data in
  let log_min = Tensor.log (Tensor.min sorted_data) in
  let log_max = Tensor.log (Tensor.max sorted_data) in
  let log_bins = Tensor.linspace log_min log_max num_bins in
  let bins = Tensor.exp log_bins in
  let indices = Tensor.bucketize sorted_data bins in
  let binned_data = Tensor.zeros [num_bins] in
  let counts = Tensor.zeros [num_bins] in
  Tensor.iteri (fun i idx ->
    let bin = Tensor.get indices [i] |> int_of_float in
    Tensor.set binned_data [bin] ((Tensor.get binned_data [bin]) +. (Tensor.get sorted_data [i]));
    Tensor.set counts [bin] ((Tensor.get counts [bin]) +. 1.0)
  ) sorted_data;
  let avg_binned_data = Tensor.div binned_data counts in
  (bins, avg_binned_data)

let safe_division a b =
  if b = 0.0 then
    (warning "Division by zero attempted"; 0.0)
  else
    a /. b

let safe_log x =
  if x <= 0.0 then
    (warning "Logarithm of non-positive number attempted"; 0.0)
  else
    log x

exception InvalidParameter of string

let validate_positive name value =
  if value <= 0.0 then
    raise (InvalidParameter (Printf.sprintf "%s must be positive" name))

let validate_non_negative name value =
  if value < 0.0 then
    raise (InvalidParameter (Printf.sprintf "%s must be non-negative" name))

let classify_regime area genus =
  if genus = 0 then Types.Planar
  else if float_of_int genus /. area > 1.0 then Types.Foamy
  else Types.HigherGenus

let exponential_fit x y =
  let log_y = Tensor.log y in
  let (a, b) = linear_regression x log_y in
  (exp b, a)

let moving_average data window_size =
  let n = List.length data in
  let result = ref [] in
  for i = 0 to n - window_size do
    let window = List.sub data i window_size in
    let avg = List.fold_left (+.) 0.0 window /. float_of_int window_size in
    result := avg :: !result
  done;
  List.rev !result

let exponential_moving_average data alpha =
  let ema = ref (List.hd data) in
  let result = ref [!ema] in
  List.iter (fun x ->
    ema := alpha *. x +. (1.0 -. alpha) *. !ema;
    result := !ema :: !result
  ) (List.tl data);
  List.rev !result