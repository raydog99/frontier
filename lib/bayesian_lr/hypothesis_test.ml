open Torch
open Type

let test_parameter samples idx null_value alpha =
  let values = List.map (fun s -> get s.theta_debiased idx) samples in
  let n = List.length values in
  
  (* Compute mean and standard error *)
  let mean = List.fold_left (+.) 0. values /. float_of_int n in
  let se = sqrt (List.fold_left (fun acc x -> 
    acc +. (x -. mean) ** 2.
  ) 0. values /. float_of_int (n - 1)) in
  
  (* Test statistic *)
  let t_stat = (mean -. null_value) /. se in
  
  (* Compute p-value (two-sided test) *)
  let p_value = 2. *. (1. -. Distribution.normal_cdf (abs_float t_stat) 0. 1.) in
  
  (* Confidence interval *)
  let z_value = Distribution.normal_ppf (1. -. alpha /. 2.) 0. 1. in
  let ci_lower = mean -. z_value *. se in
  let ci_upper = mean +. z_value *. se in
  
  {
    parameter_idx = idx;
    null_value;
    p_value;
    test_statistic = t_stat;
    ci_lower;
    ci_upper;
  }

let test_joint samples indices null_values alpha =
  (* Extract relevant components *)
  let n_params = List.length indices in
  let selected_samples = List.map (fun s ->
    let selected = zeros [n_params] in
    List.iteri (fun i idx ->
      tensor_set selected [i] (get s.theta_debiased idx)
    ) indices;
    selected
  ) samples in
  
  (* Compute mean and precision matrix *)
  let mean = List.fold_left (fun acc s -> add acc s) (zeros [n_params]) selected_samples in
  let mean = scalar_mul (1. /. float_of_int (List.length samples)) mean in
  
  let null_tensor = of_float1 null_values in
  let diff = sub mean null_tensor in
  
  (* Compute precision matrix *)
  let centered = List.map (fun s -> sub s mean) selected_samples in
  let precision = List.fold_left (fun acc s ->
    let outer = matmul (unsqueeze s 1) (unsqueeze s 0) in
    add acc outer
  ) (zeros [n_params; n_params]) centered in
  let precision = scalar_mul (1. /. float_of_int (List.length samples)) precision in
  
  (* Compute test statistic (Mahalanobis distance) *)
  let test_stat = dot diff (matmul precision diff) |> float_of_elt in
  
  (* p-value from chi-square distribution *)
  let p_value = 1. -. Distribution.chi_square_cdf test_stat (float_of_int n_params) in
  
  {indices; null_values; test_statistic = test_stat; p_value}