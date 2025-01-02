open Torch
open Types
open Utils

let objective_function (weights: Tensor.t) (expected_returns: Tensor.t) (cov_matrix: Tensor.t) (risk_aversion: float) : float =
  let portfolio_return = Tensor.dot weights expected_returns |> Tensor.item in
  let portfolio_risk = Tensor.matmul weights (Tensor.matmul cov_matrix weights) |> Tensor.item in
  portfolio_return -. (risk_aversion *. portfolio_risk)

let optimize_subproblem (subproblem: portfolio) (cov_matrix: covariance_matrix) (params: optimization_params) : optimization_result =
  let n = Array.length subproblem.assets in
  let weights = Tensor.ones [n] |> Tensor.div_scalar (Scalar.float (float_of_int n)) in
  let expected_returns = subproblem.expected_returns in
  
  let rec optimize weights iter =
    if iter >= params.max_iterations then
      raise (OptimizationError "Maximum iterations reached")
    else
      let obj_value = objective_function weights expected_returns cov_matrix params.risk_aversion in
      let gradient = Tensor.sub (Tensor.matmul cov_matrix weights |> Tensor.mul_scalar (Scalar.float (2.0 *. params.risk_aversion)))
                                expected_returns in
      let new_weights = Tensor.sub weights (Tensor.mul_scalar gradient (Scalar.float 0.01)) in
      let new_weights = Tensor.clamp new_weights ~min:(Scalar.float 0.0) ~max:(Scalar.float 1.0) in
      let new_weights = Tensor.div new_weights (Tensor.sum new_weights) in
      
      (match params.cardinality_constraint with
      | Some k ->
          let _, top_k_indices = Tensor.topk new_weights k ~largest:true ~sorted:false in
          let mask = Tensor.zeros_like new_weights in
          Tensor.index_fill_ mask 0 top_k_indices (Scalar.float 1.0);
          Tensor.mul_inplace new_weights mask;
          Tensor.div_inplace new_weights (Tensor.sum new_weights)
      | None -> ());
      
      if Tensor.allclose weights new_weights ~rtol:params.tolerance ~atol:params.tolerance then
        { weights = new_weights; objective_value = obj_value; iterations = iter }
      else
        optimize new_weights (iter + 1)
  in
  
  try
    log_message "Starting subproblem optimization";
    let result = optimize weights 0 in
    log_message (Printf.sprintf "Subproblem optimization completed in %d iterations" result.iterations);
    result
  with
  | OptimizationError msg ->
      log_message ("Optimization error: " ^ msg);
      raise (OptimizationError msg)

let aggregate_solutions (subproblems: portfolio array) (optimization_results: optimization_result array) : portfolio =
  let total_assets = Array.fold_left (fun acc p -> acc + Array.length p.assets) 0 subproblems in
  let aggregated_assets = Array.make total_assets subproblems.(0).assets.(0) in
  let aggregated_weights = Tensor.zeros [total_assets] in
  let aggregated_returns = Tensor.zeros [total_assets] in
  
  let idx = ref 0 in
  Array.iteri (fun i subproblem ->
    Array.iteri (fun j asset ->
      aggregated_assets.(!idx) <- asset;
      Tensor.set aggregated_weights [!idx] (Tensor.get optimization_results.(i).weights [j]);
      Tensor.set aggregated_returns [!idx] (Tensor.get subproblem.expected_returns [j]);
      incr idx;
    ) subproblem.assets;
  ) subproblems;
  
  let aggregated_portfolio = {
    assets = aggregated_assets;
    weights = aggregated_weights;
    expected_returns = aggregated_returns;
  } in
  
  validate_portfolio aggregated_portfolio;
  aggregated_portfolio