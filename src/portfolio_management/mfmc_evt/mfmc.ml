open Torch
open Types

let baseline_estimator (type p) (module D : Distribution with type params = p) x_high =
    D.fit_mle x_high

let joint_mle (type p) (module D : Distribution with type params = p) x_high x_low =
    let x_combined = Tensor.cat [x_high; x_low] ~dim:1 in
    D.fit_mle x_combined

let moment_mf (type p) 
      (module D : Distribution with type params = p) x_high x_low alpha =
    let params_high = D.fit_mle x_high |> Result.get_ok in
    let params_low = D.fit_mle x_low |> Result.get_ok in
    let mean_high = Tensor.mean x_high in
    let mean_low = Tensor.mean x_low in
    Ok (D.params_from_tensor Tensor.(D.params_to_tensor params_high |> Result.get_ok + 
        alpha * (D.params_to_tensor params_low |> Result.get_ok - mean_low)) |> Result.get_ok)

let marginal_mle (type p) 
      (module D : Distribution with type params = p) x_high x_low gamma =
    let params_high = D.fit_mle x_high |> Result.get_ok in
    let params_low = D.fit_mle x_low |> Result.get_ok in
    Ok (D.params_from_tensor Tensor.(D.params_to_tensor params_high |> Result.get_ok + 
        gamma * (D.params_to_tensor params_low |> Result.get_ok - D.params_to_tensor params_high |> Result.get_ok)) |> Result.get_ok)

let optimal_alpha high_fidelity low_fidelity n =
    let x_high = high_fidelity n in
    let x_low = low_fidelity n in
    let cov = Tensor.cov x_high x_low in
    let var_low = Tensor.var x_low in
    Ok Tensor.(cov / var_low)

let asymptotic_variance (type p) 
      (module D : Distribution with type params = p) params n =
    let x = D.sample params n |> Result.get_ok in
    let log_likelihood = D.log_likelihood params x |> Result.get_ok in
    let grad = Tensor.grad log_likelihood in
    let hessian = Tensor.hessian log_likelihood in
    Tensor.(neg (mean hessian) |> inverse |> diag)

let compare_estimators (type p)
      (module D : Distribution with type params = p)
      true_params n m num_trials =
    let baseline_mse = Tensor.zeros [5] in
    let joint_mle_mse = Tensor.zeros [5] in
    let moment_mf_mse = Tensor.zeros [5] in
    let marginal_mle_mse = Tensor.zeros [5] in
    for _ = 1 to num_trials do
      let x_high = D.sample true_params n |> Result.get_ok in
      let x_low = D.sample true_params (n + m) |> Result.get_ok in
      let baseline_est = baseline_estimator (module D) x_high |> Result.get_ok in
      let joint_mle_est = joint_mle (module D) x_high x_low |> Result.get_ok in
      let alpha = optimal_alpha (D.sample true_params) (D.sample true_params) n |> Result.get_ok in
      let moment_mf_est = moment_mf (module D) x_high x_low alpha |> Result.get_ok in
      let gamma = optimal_alpha (D.sample true_params) (D.sample true_params) n |> Result.get_ok in
      let marginal_mle_est = marginal_mle (module D) x_high x_low gamma |> Result.get_ok in
      Tensor.(baseline_mse += pow (D.params_to_tensor baseline_est |> Result.get_ok - D.params_to_tensor true_params |> Result.get_ok) (scalar 2));
      Tensor.(joint_mle_mse += pow (D.params_to_tensor joint_mle_est |> Result.get_ok - D.params_to_tensor true_params |> Result.get_ok) (scalar 2));
      Tensor.(moment_mf_mse += pow (D.params_to_tensor moment_mf_est |> Result.get_ok - D.params_to_tensor true_params |> Result.get_ok) (scalar 2));
      Tensor.(marginal_mle_mse += pow (D.params_to_tensor marginal_mle_est |> Result.get_ok - D.params_to_tensor true_params |> Result.get_ok) (scalar 2));
    done;
    Tensor.(baseline_mse /= scalar (float_of_int num_trials));
    Tensor.(joint_mle_mse /= scalar (float_of_int num_trials));
    Tensor.(moment_mf_mse /= scalar (float_of_int num_trials));
    Tensor.(marginal_mle_mse /= scalar (float_of_int num_trials));
    (baseline_mse, joint_mle_mse, moment_mf_mse, marginal_mle_mse)

module QoI = struct
  let exceedance_probability params x =
    match params with
    | GumbelParams _ ->
      let open Tensor in
      Ok (float 1. - (Gumbel.log_likelihood params x |> Result.get_ok |> exp))
    | _ -> Error (InvalidParameters "QoI.exceedance_probability: Invalid parameters")

  let extreme_quantile params p =
    match params with
    | GumbelParams { mu; beta } ->
      Ok Tensor.(float mu - float beta * log (neg (log (float p))))
    | _ -> Error (InvalidParameters "QoI.extreme_quantile: Invalid parameters")
end