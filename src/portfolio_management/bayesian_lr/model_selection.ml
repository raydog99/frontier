open Torch
open Type

let get_support theta =
  map (fun x -> if abs_float x > 1e-6 then 1. else 0.) theta

let compute_tpr true_support pred_support =
  let tp = dot true_support pred_support in
  let p = sum true_support in
  tp /. p

let compute_fdp true_support pred_support =
  let fp = dot (sub (ones_like true_support) true_support) pred_support in
  let total_pos = sum pred_support in
  if total_pos > 0. then fp /. total_pos else 0.

let compute_mcc true_support pred_support =
  let tp = dot true_support pred_support in
  let tn = dot (sub (ones_like true_support) true_support) 
               (sub (ones_like pred_support) pred_support) in
  let fp = dot (sub (ones_like true_support) true_support) pred_support in
  let fn = dot true_support (sub (ones_like pred_support) pred_support) in
  
  let numerator = tp *. tn -. fp *. fn in
  let denominator = sqrt ((tp +. fp) *. (tp +. fn) *. (tn +. fp) *. (tn +. fn)) in
  
  if denominator > 0. then numerator /. denominator else 0.

let evaluate_selection samples true_theta =
  let true_support = get_support true_theta in
  
  (* Compute median probability model *)
  let n_samples = List.length samples in
  let support_sum = List.fold_left 
    (fun acc s -> add acc (get_support s.theta_star))
    (zeros_like true_support)
    samples in
  
  let median_support = map 
    (fun x -> if x > (float_of_int n_samples) /. 2. then 1. else 0.)
    support_sum in
  
  {
    tpr = compute_tpr true_support median_support;
    fdp = compute_fdp true_support median_support;
    mcc = compute_mcc true_support median_support;
  }

let get_top_variables samples k =
  let n_samples = List.length samples in
  let p = size (List.hd samples).theta 0 in
  
  (* Count selection frequency for each variable *)
  let freq = zeros [p] in
  List.iter (fun s ->
    let support = get_support s.theta_star in
    tensor_add_ freq support
  ) samples;
  
  (* Get indices of top k most frequently selected variables *)
  let _, indices = sort freq ~descending:true in
  narrow indices 0 0 k