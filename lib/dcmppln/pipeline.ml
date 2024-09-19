open Torch
open Types
open Preprocessing
open Clustering
open Optimization

let run_decomposition_pipeline (portfolio: portfolio) (num_communities: int) (params: optimization_params) : portfolio =
  try
    log_message "Starting decomposition pipeline";
    validate_portfolio portfolio;
    
    let returns = Tensor.stack (Array.map (fun asset -> asset.returns) portfolio.assets) ~dim:0 in
    let n, t = Tensor.shape2_exn returns in
    
    log_message "Computing covariance and correlation matrices";
    let cov_matrix = compute_covariance_matrix returns in
    let corr_matrix = compute_correlation_matrix cov_matrix in
    let cleaned_corr_matrix = preprocess_correlation_matrix corr_matrix n t in
    
    log_message "Performing spectral clustering";
    let communities = spectral_clustering cleaned_corr_matrix num_communities in
    let subproblems = decompose_portfolio portfolio communities in
    
    log_message "Optimizing subproblems";
    let optimization_results = Array.map (fun subproblem ->
      let sub_cov_matrix = Tensor.index_select cov_matrix 0 (float_array_to_tensor (Array.map float_of_int (Array.init (Array.length subproblem.assets) (fun i -> i)))) in
      let sub_cov_matrix = Tensor.index_select sub_cov_matrix 1 (float_array_to_tensor (Array.map float_of_int (Array.init (Array.length subproblem.assets) (fun i -> i)))) in
      optimize_subproblem subproblem sub_cov_matrix params
    ) subproblems in
    
    log_message "Aggregating solutions";
    let result = aggregate_solutions subproblems optimization_results in
    validate_portfolio result;
    
    log_message "Decomposition pipeline completed successfully";
    result
  with
  | OptimizationError msg ->
      log_message ("Optimization error: " ^ msg);
      failwith ("Optimization error: " ^ msg)
  | e ->
      log_message ("Unexpected error: " ^ Printexc.to_string e);
      raise e