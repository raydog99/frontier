open Torch

type convergence_stats = {
  potential_scale_reduction: float;
  effective_sample_size: float;
  autocorrelation_time: float;
  spectral_gap_estimate: float;
}

let gelman_rubin chains burn_in =
  let n_chains = List.length chains in
  let chain_length = Tensor.size (List.hd chains) 0 - burn_in in
  
  let chain_means = List.map (fun chain ->
    let chain = Tensor.narrow chain 0 burn_in chain_length in
    Tensor.mean chain ~dim:[0] ~keepdim:false
  ) chains in
  
  let overall_mean = 
    Tensor.mean (Tensor.stack chain_means ~dim:0) ~dim:[0] ~keepdim:false in
  
  let b = float_of_int chain_length *.
    (List.fold_left (fun acc mean ->
      acc +. Tensor.norm 
        (Tensor.sub mean overall_mean) 
        ~p:(Scalar 2) 
        ~dim:[0] 
        ~keepdim:false
        |> Tensor.item
    ) 0. chain_means) /. float_of_int (n_chains - 1) in
  
  let w = List.fold_left (fun acc chain ->
    let chain = Tensor.narrow chain 0 burn_in chain_length in
    let centered = Tensor.sub chain 
      (Tensor.expand_as 
        (Tensor.mean chain ~dim:[0] ~keepdim:true) 
        chain) in
    acc +. Tensor.sum (Tensor.pow centered 2.0) |> Tensor.item
  ) 0. chains /. float_of_int (n_chains * (chain_length - 1)) in
  
  let var_plus = ((float_of_int (chain_length - 1) *. w +. b) /. 
                  float_of_int chain_length) in
  sqrt (var_plus /. w)

let effective_sample_size samples max_lag =
  let n = Tensor.size samples 0 in
  
  let auto_corr = Array.init max_lag (fun lag ->
    let x1 = Tensor.narrow samples 0 0 (n - lag) in
    let x2 = Tensor.narrow samples 0 lag (n - lag) in
    let corr = Tensor.mean (Tensor.mul x1 x2) ~dim:[0] ~keepdim:false in
    Tensor.item corr
  ) in
  
  let rec find_cutoff i =
    if i >= max_lag || auto_corr.(i) < 0. then i
    else find_cutoff (i + 1)
  in
  let cutoff = find_cutoff 1 in
  
  let sum_rho = Array.fold_left (+.) 0. 
    (Array.sub auto_corr 1 (cutoff - 1)) in
  float_of_int n /. (1. +. 2. *. sum_rho)

let analyze_convergence samples burn_in =
  let psrf = gelman_rubin samples burn_in in
  let main_chain = List.hd samples in
  let ess = effective_sample_size main_chain 50 in
  let act = float_of_int (Tensor.size main_chain 0) /. ess in
  let gap = 1. /. act in
  
  {
    potential_scale_reduction = psrf;
    effective_sample_size = ess;
    autocorrelation_time = act;
    spectral_gap_estimate = gap;
  }

let has_converged stats threshold =
  stats.potential_scale_reduction <= 1.1 &&
  stats.effective_sample_size >= threshold