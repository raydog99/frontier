open Torch
open Types
open Distributions
open Mfmc

(* Parametric quantities of interest *)
let estimate_qoi (type p) 
    (module D : Distribution with type params = p) 
    qoi_func params_est =
  qoi_func params_est

let qoi_asymptotic_variance (type p) 
    (module D : Distribution with type params = p) 
    qoi_func params n =
  let params_tensor = D.params_to_tensor params |> Result.get_ok in
  let qoi = qoi_func params in
  let grad_qoi = Tensor.grad qoi in
  let avar_params = MFMC.asymptotic_variance (module D) params n in
  Tensor.(grad_qoi * avar_params * grad_qoi)

let fit_regression x_high x_low =
  let open Tensor in
  let x = stack [x_high; x_low] ~dim:1 in
  let y = x_high in
  let xt = transpose x ~dim0:0 ~dim1:1 in
  let beta = matmul (matmul (inverse (matmul xt x)) xt) y in
  let alpha = get beta 0 0 |> float_value in
  let gamma = get beta 1 0 |> float_value in
  alpha, gamma

let predict_regression alpha gamma x_low =
  Tensor.(float alpha + float gamma * x_low)

let regression_mf_estimate x_high x_low x_low_extra =
  let alpha, gamma = fit_regression x_high x_low in
  let y_pred = predict_regression alpha gamma x_low_extra in
  Tensor.(mean x_high + gamma * (mean x_low_extra - mean x_low))

let importance_sampling (type p)
    (module D : Distribution with type params = p)
    proposal_dist target_dist n =
  let samples = D.sample proposal_dist n |> Result.get_ok in
  let weights = Tensor.(exp (D.log_likelihood target_dist samples |> Result.get_ok - 
                             D.log_likelihood proposal_dist samples |> Result.get_ok)) in
  let normalized_weights = Tensor.(weights / sum weights) in
  samples, normalized_weights

let analyze_bivariate_gaussian () =
  let true_params = BivariateGaussianParams { mu1 = 0.; mu2 = 0.; sigma1 = 1.; sigma2 = 1.; rho = 0.5 } in
  let n = 100 in
  let m = 1000 in
  let num_trials = 1000 in
  let (baseline_mse, joint_mle_mse, moment_mf_mse, marginal_mle_mse) =
    MFMC.compare_estimators (module BivariateGaussian) true_params n m num_trials in
  
  Printf.printf "Bivariate Gaussian Analysis Results:\n";
  Printf.printf "Baseline MSE: %s\n" (Tensor.to_string baseline_mse);
  Printf.printf "Joint MLE MSE: %s\n" (Tensor.to_string joint_mle_mse);
  Printf.printf "Moment MF MSE: %s\n" (Tensor.to_string moment_mf_mse);
  Printf.printf "Marginal MLE MSE: %s\n" (Tensor.to_string marginal_mle_mse)

let analyze_bivariate_gumbel () =
  let true_params = BivariateGumbelParams { mu1 = 0.; mu2 = 0.; beta1 = 1.; beta2 = 1.; r = 0.5 } in
  let n = 100 in
  let m = 1000 in
  let num_trials = 1000 in
  let (baseline_mse, joint_mle_mse, moment_mf_mse, marginal_mle_mse) =
    MFMC.compare_estimators (module BivariateGumbel) true_params n m num_trials in
  
  Printf.printf "Bivariate Gumbel Analysis Results:\n";
  Printf.printf "Baseline MSE: %s\n" (Tensor.to_string baseline_mse);
  Printf.printf "Joint MLE MSE: %s\n" (Tensor.to_string joint_mle_mse);
  Printf.printf "Moment MF MSE: %s\n" (Tensor.to_string moment_mf_mse);
  Printf.printf "Marginal MLE MSE: %s\n" (Tensor.to_string marginal_mle_mse)

let analyze_binary_outcomes () =
  let true_params = GaussianParams { mu = 0.7; sigma = 0. } in
  let n = 100 in
  let m = 1000 in
  let num_trials = 1000 in
  let (baseline_mse, joint_mle_mse, moment_mf_mse, marginal_mle_mse) =
    MFMC.compare_estimators (module Bernoulli) true_params n m num_trials in
  
  Printf.printf "Binary Outcomes Analysis Results:\n";
  Printf.printf "Baseline MSE: %s\n" (Tensor.to_string baseline_mse);
  Printf.printf "Joint MLE MSE: %s\n" (Tensor.to_string joint_mle_mse);
  Printf.printf "Moment MF MSE: %s\n" (Tensor.to_string moment_mf_mse);
  Printf.printf "Marginal MLE MSE: %s\n" (Tensor.to_string marginal_mle_mse)