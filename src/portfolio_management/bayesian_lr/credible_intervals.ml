open Torch
open Type

let compute_component_intervals samples j alpha =
  let values = List.map (fun s -> get s.theta_debiased idx) samples |> Array.of_list in
  Array.sort compare values;
  let n = Array.length values in
  let lower_idx = int_of_float (float_of_int n *. alpha /. 2.) in
  let upper_idx = int_of_float (float_of_int n *. (1. -. alpha /. 2.)) in
  let median_idx = n / 2 in
  
  {
    lower = values.(lower_idx);
    upper = values.(upper_idx);
    median = values.(median_idx);
    coverage = 1. -. alpha;
  }

let compute_credible_ellipsoid samples selected_vars alpha =
  let n_samples = List.length samples in
  let p = List.length selected_vars in
  
  (* Get selected components from samples *)
  let selected_samples = List.map (fun s ->
    let selected = zeros [p] in
    List.iteri (fun i j ->
      tensor_set selected [i] (get s.theta_star j)
    ) selected_vars;
    selected
  ) samples in
  
  (* Compute empirical mean and precision *)
  let mean = List.fold_left (fun acc s -> add acc s) (zeros [p]) selected_samples in
  let mean = scalar_mul (1. /. float_of_int n_samples) mean in
  
  let centered = List.map (fun s -> sub s mean) selected_samples in
  let precision = List.fold_left (fun acc s ->
    let outer = matmul (unsqueeze s 1) (unsqueeze s 0) in
    add acc outer
  ) (zeros [p; p]) centered in
  let precision = scalar_mul (1. /. float_of_int n_samples) precision in
  
  (* Compute radius for desired coverage *)
  let distances = List.map (fun s ->
    let diff = sub s mean in
    let quad = dot diff (matmul precision diff) in
    float_of_elt quad
  ) selected_samples in
  Array.sort compare (Array.of_list distances);
  let radius = List.nth (List.sort compare distances) 
    (int_of_float (float_of_int n_samples *. (1. -. alpha))) in
  
  {center = mean; precision; radius}