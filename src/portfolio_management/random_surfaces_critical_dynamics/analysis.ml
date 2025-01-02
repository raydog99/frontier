open Torch
open Types
open Constants
open Utils
open Logging

let analyze_regime_transitions areas genera =
  debug "Analyzing regime transitions";
  try
    let transitions = ref [] in
    let current_regime = ref (classify_regime (List.hd areas) (List.hd genera)) in
    List.iter2 (fun area genus ->
      let new_regime = classify_regime area genus in
      if new_regime <> !current_regime then begin
        transitions := (new_regime, area, genus) :: !transitions;
        current_regime := new_regime
      end
    ) areas genera;
    List.rev !transitions
  with
  | e ->
      error (Printf.sprintf "Error during regime transition analysis: %s" (Printexc.to_string e));
      raise e

let compute_autocorrelation returns max_lag =
  debug "Computing autocorrelation";
  try
    let n = Tensor.shape returns |> List.hd in
    let mean = Tensor.mean returns in
    let variance = Tensor.var returns in
    let normalized_returns = Tensor.sub returns mean in
    let acf = Tensor.zeros [max_lag + 1] in
    for lag = 0 to max_lag do
      let product = Tensor.mul (Tensor.narrow normalized_returns 0 0 (n - lag))
                               (Tensor.narrow normalized_returns 0 lag (n - lag)) in
      let cov = Tensor.sum product /. float_of_int (n - lag) in
      Tensor.set acf [lag] (cov /. variance)
    done;
    acf
  with
  | e ->
      error (Printf.sprintf "Error during autocorrelation computation: %s" (Printexc.to_string e));
      raise e

let estimate_lyapunov_exponent time_series =
  debug "Estimating Lyapunov exponent";
  try
    let n = List.length time_series - 1 in
    let diffs = List.map2 (fun x y -> abs_float (log (abs_float (y /. x)))) 
                          (List.take n time_series) 
                          (List.drop 1 time_series) in
    List.fold_left (+.) 0.0 diffs /. float_of_int n
  with
  | e ->
      error (Printf.sprintf "Error during Lyapunov exponent estimation: %s" (Printexc.to_string e));
      raise e

let compute_fractal_dimension returns num_boxes =
  debug "Computing fractal dimension";
  try
    let min_return = Tensor.min returns in
    let max_return = Tensor.max returns in
    let box_size = (max_return -. min_return) /. float_of_int num_boxes in
    let boxes = Tensor.zeros [num_boxes] in
    Tensor.iter (fun r ->
      let box_index = int_of_float ((r -. min_return) /. box_size) in
      Tensor.set boxes [box_index] (Tensor.get boxes [box_index] +. 1.0)
    ) returns;
    let non_empty_boxes = Tensor.sum (Tensor.gt boxes (Tensor.of_float 0.0)) in
    log (non_empty_boxes) /. log (1.0 /. float_of_int num_boxes)
  with
  | e ->
      error (Printf.sprintf "Error during fractal dimension computation: %s" (Printexc.to_string e));
      raise e