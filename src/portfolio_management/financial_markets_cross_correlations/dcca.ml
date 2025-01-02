open Torch

let integrate_series x =
  let open Tensor in
  let mean = mean x in
  cumsum (sub x mean) ~dim:0

let detrend_series integrated_series s =
  let open Tensor in
  let n = shape integrated_series |> List.hd in
  let num_boxes = n / s in
  let detrended = Tensor.zeros [n] in
  
  for i = 0 to num_boxes - 1 do
    let start_idx = i * s in
    let end_idx = min (start_idx + s) n in
    let box = slice integrated_series start_idx end_idx in
    let t = Tensor.arange ~start:0 ~end_:(Float.of_int (end_idx - start_idx)) ~options:(kind Float, device CPU) in
    let coeffs = (lstsq box t).coefficients in
    let trend = add (mul t (get coeffs 0)) (get coeffs 1) in
    let detrended_box = sub box trend in
    set detrended ~start_idx ~end_idx detrended_box
  done;
  detrended

let dcca_coefficient x y s =
  let integrated_x = integrate_series x in
  let integrated_y = integrate_series y in
  let detrended_x = detrend_series integrated_x s in
  let detrended_y = detrend_series integrated_y s in
  
  let f2_dcca = Tensor.mean (Tensor.mul detrended_x detrended_y) in
  let f2_dfa_x = Tensor.mean (Tensor.mul detrended_x detrended_x) in
  let f2_dfa_y = Tensor.mean (Tensor.mul detrended_y detrended_y) in
  
  let coefficient = Tensor.div f2_dcca (Tensor.sqrt (Tensor.mul f2_dfa_x f2_dfa_y)) in
  Tensor.to_float0_exn coefficient

let dcca_distance coefficient =
  sqrt (2.0 *. (1.0 -. (coefficient *. coefficient)))