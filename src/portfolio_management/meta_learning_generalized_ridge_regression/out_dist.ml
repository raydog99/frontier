open Torch
open Matrix_ops

let analyze_ood_prediction ~omega ~omega_hat ~sigma ~psi ~epsilon =
  let dim = Tensor.size omega |> List.hd in
  
  (* Compute condition numbers *)
  let cond_omega = 
    let eigs = Tensor.symeig omega ~eigenvectors:false in
    let max_eval = Tensor.get eigs [|dim-1|] |> Tensor.to_float0_exn in
    let min_eval = Tensor.get eigs [|0|] |> Tensor.to_float0_exn in
    max_eval /. min_eval in
  
  let cond_psi =
    let eigs = Tensor.symeig psi ~eigenvectors:false in
    let max_eval = Tensor.get eigs [|dim-1|] |> Tensor.to_float0_exn in
    let min_eval = Tensor.get eigs [|0|] |> Tensor.to_float0_exn in
    max_eval /. min_eval in
  
  (* Compute operator norm differences *)
  let diff_norm = operator_norm (Tensor.sub omega_hat omega) in
  
  (* Bound terms *)
  let term1 = epsilon *. cond_omega *. cond_psi *. 
              operator_norm sigma *. diff_norm in
  let term2 = operator_norm sigma *. 
              (1. +. cond_omega +. cond_psi) *. 
              cond_omega *. cond_psi *. diff_norm in
  
  (term1, term2)

let compute_risk_bound ~omega ~omega_hat ~sigma ~gamma ~epsilon =
  let dim = Tensor.size omega |> List.hd in
  
  (* Compute terms from risk bound *)
  let diff = Tensor.sub omega_hat omega in
  let diff_norm = operator_norm diff in
  
  (* Compute constants *)
  let c_omega = operator_norm omega *. 
                operator_norm (Tensor.inverse omega) in
  let c_sigma = operator_norm sigma in
  
  (* Compute bound *)
  let bound = gamma *. epsilon *. c_omega *. c_sigma *. diff_norm *. 
              (2. +. c_omega) in
  
  bound

let verify_ood_conditions ~omega ~psi ~epsilon =
  let dim = Tensor.size omega |> List.hd in
  
  (* Verify positive definiteness *)
  let is_pd_omega = is_positive_definite omega in
  let is_pd_psi = is_positive_definite psi in
  
  (* Verify bounded difference *)
  let diff = Tensor.sub psi omega in
  let diff_norm = operator_norm diff in
  let bound_satisfied = diff_norm <= epsilon in
  
  is_pd_omega && is_pd_psi && bound_satisfied