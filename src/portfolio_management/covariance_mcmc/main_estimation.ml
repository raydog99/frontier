open Torch
open Types

type config = {
  device: Gpu_compute.device_config;
  regularization: Ill_conditioned.regularization;
  batch_config: Batch_processing.batch_config;
  n_parallel_chains: int;
}

let estimate_covariance chain initial n_samples config =
  (* Generate samples using parallel chains *)
  let samples = Parallel_processing.parallel_chain_execution
    chain 
    initial 
    config.n_parallel_chains 
    n_samples 
    config.device in
  
  (* Combine samples and check convergence *)
  let all_samples = Tensor.cat (Array.of_list samples) ~dim:0 in
  let conv_stats = Convergence_diagnostics.analyze_convergence
    samples (n_samples / 10) in
  
  (* Compute robust covariance estimate *)
  let dist = Ill_conditioned.robust_covariance_estimation
    all_samples 
    config.regularization 
    config.device in
  
  dist, conv_stats.potential_scale_reduction

let estimate_with_guarantees chain initial n_samples config =
  let dist, psrf = estimate_covariance chain initial n_samples config in
  
  (* Generate new samples for verification *)
  let samples = Parallel_processing.parallel_chain_execution
    chain 
    initial 
    config.n_parallel_chains 
    n_samples 
    config.device in
  
  let all_samples = Tensor.cat (Array.of_list samples) ~dim:0 in
  
  (* Check convergence *)
  let conv_stats = Convergence_diagnostics.analyze_convergence
    samples (n_samples / 10) in
  
  (* Verify guarantees *)
  let guarantees = Bounds.all_bounds
    all_samples 
    chain 
    1.0   (* c_pi *)
    0.1   (* epsilon *)
    0.1   (* delta *)
    config.device in
  
  dist, conv_stats, guarantees