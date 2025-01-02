open Torch
open Utils

(* Normal space density rescaling *)
let rescale_transition alpha p =
  let power = alpha /. (alpha +. 1.) in
  Tensor.(pow p (Scalar.f power)) |> normalize
  
let rescale_measurement beta p =
  let power = beta /. (beta +. 1.) in
  Tensor.(pow p (Scalar.f power)) |> normalize
  
(* Log space density rescaling for numerical stability *)
let rescale_transition_log alpha log_p =
  let power = alpha /. (alpha +. 1.) in
  let scaled = Tensor.(scalar_tensor power * log_p) in
  log_normalize scaled
  
let rescale_measurement_log beta log_p =
  let power = beta /. (beta +. 1.) in
  let scaled = Tensor.(scalar_tensor power * log_p) in
  log_normalize scaled
  
(* Information bottleneck optimization *)
let optimize_bottleneck ~prior ~transition_prob ~measurement_prob ~lambda =
  (* Compute mutual information between distributions *)
  let compute_mutual_info p_joint p_marginal =
    let p_ratio = Tensor.(p_joint / p_marginal) in
    let mi = Tensor.(mean (p_joint * log p_ratio)) in
    mi
  in
  
  (* Optimal solution using density rescaling *)
  let gamma = lambda /. (1. +. lambda) in
  let q_info = Tensor.(
    prior * (pow measurement_prob (Scalar.f gamma))
  ) |> normalize in
  
  q_info