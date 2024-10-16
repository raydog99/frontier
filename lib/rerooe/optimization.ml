open Torch

let mean_variance cov_matrix expected_returns =
  let n = Tensor.shape cov_matrix |> List.hd in
  let ones = Tensor.ones [1; n] in
  let inv_cov = Tensor.inverse cov_matrix in
  let a = Tensor.matmul (Tensor.matmul ones inv_cov) (Tensor.transpose ones ~dim0:0 ~dim1:1) |> Tensor.to_float0_exn in
  let b = Tensor.matmul (Tensor.matmul ones inv_cov) (Tensor.transpose expected_returns ~dim0:0 ~dim1:1) |> Tensor.to_float0_exn in
  let c = Tensor.matmul (Tensor.matmul expected_returns inv_cov) (Tensor.transpose expected_returns ~dim0:0 ~dim1:1) |> Tensor.to_float0_exn in
  let lambda = 2.0 in (* Risk aversion parameter *)
  let weights = Tensor.matmul inv_cov (Tensor.transpose (Tensor.add (Tensor.mul_scalar expected_returns (1. /. lambda)) (Tensor.mul_scalar ones ((a -. b /. lambda) /. a))) ~dim0:0 ~dim1:1) in
  Tensor.to_float1 weights

let minimum_variance cov_matrix =
  let n = Tensor.shape cov_matrix |> List.hd in
  let ones = Tensor.ones [1; n] in
  let inv_cov = Tensor.inverse cov_matrix in
  let weights = Tensor.matmul inv_cov (Tensor.transpose ones ~dim0:0 ~dim1:1) in
  let sum_weights = Tensor.sum weights in
  let normalized_weights = Tensor.div weights sum_weights in
  Tensor.to_float1 normalized_weights

let maximum_sharpe_ratio cov_matrix expected_returns risk_free_rate =
  let n = Tensor.shape cov_matrix |> List.hd in
  let excess_returns = Tensor.sub expected_returns (Tensor.full [1; n] risk_free_rate) in
  let inv_cov = Tensor.inverse cov_matrix in
  let weights = Tensor.matmul inv_cov (Tensor.transpose excess_returns ~dim0:0 ~dim1:1) in
  let sum_weights = Tensor.sum weights in
  let normalized_weights = Tensor.div weights sum_weights in
  Tensor.to_float1 normalized_weights

let risk_parity cov_matrix =
  let n = Tensor.shape cov_matrix |> List.hd in
  let initial_weights = Tensor.full [1; n] (1. /. float_of_int n) in
  let learning_rate = 0.01 in
  let num_iterations = 1000 in
  
  let rec optimize weights iter =
    if iter = num_iterations then weights
    else
      let risk_contributions = Tensor.sqrt (Tensor.matmul (Tensor.matmul weights cov_matrix) (Tensor.transpose weights ~dim0:0 ~dim1:1)) in
      let total_risk = Tensor.sum risk_contributions in
      let target_risk = Tensor.div total_risk (float_of_int n |> Tensor.of_float0) in
      let loss = Tensor.sum (Tensor.pow (Tensor.sub risk_contributions target_risk) (Tensor.of_float0 2.)) in
      let grad = Tensor.grad loss [weights] in
      let updated_weights = Tensor.sub weights (Tensor.mul_scalar (List.hd grad) learning_rate) in
      let normalized_weights = Tensor.div updated_weights (Tensor.sum updated_weights) in
      optimize normalized_weights (iter + 1)
  in
  
  let final_weights = optimize initial_weights 0 in
  Tensor.to_float1 final_weights

let constrained_optimization objective_fn constraint_fns initial_weights =
  let n = Array.length initial_weights in
  let initial_tensor = Tensor.of_float1 initial_weights in
  
  let rec optimize weights iter =
    if iter = 1000 then weights
    else
      let weights_float = Tensor.to_float1 weights in
      if Array.for_all (fun constraint_fn -> constraint_fn weights_float) constraint_fns then
        let loss = Tensor.of_float0 (objective_fn weights_float) in
        let grad = Tensor.grad loss [weights] in
        let updated_weights = Tensor.sub weights (Tensor.mul_scalar (List.hd grad) 0.01) in
        let clamped_weights = Tensor.clamp updated_weights ~min:(Tensor.of_float0 0.) ~max:(Tensor.of_float0 1.) in
        let normalized_weights = Tensor.div clamped_weights (Tensor.sum clamped_weights) in
        optimize normalized_weights (iter + 1)
      else
        optimize (Tensor.of_float1 initial_weights) (iter + 1)
  in
  
  let final_weights = optimize initial_tensor 0 in
  Tensor.to_float1 final_weights