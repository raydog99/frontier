open Torch
open Type

let compute_acf series max_lag =
  let n = Array.length series in
  let mean = Array.fold_left (+.) 0. series /. float_of_int n in
  let centered = Array.map (fun x -> x -. mean) series in
  
  let variance = Array.fold_left (fun acc x -> acc +. x *. x) 0. centered in
  let variance = variance /. float_of_int n in
  
  Array.init max_lag (fun lag ->
    let sum = ref 0. in
    for i = 0 to n - lag - 1 do
      sum := !sum +. centered.(i) *. centered.(i + lag)
    done;
    !sum /. (float_of_int n *. variance)
  )

let compute_ess series =
  let acf = compute_acf series 50 in
  let sum_rho = Array.fold_left (+.) 1. 
    (Array.map (fun x -> if x > 0. then 2. *. x else 0.) acf) in
  float_of_int (Array.length series) /. sum_rho

let compute_psrf chains =
  let n_chains = Array.length chains in
  let chain_length = Array.length chains.(0) in
  
  (* Compute chain means *)
  let chain_means = Array.map (fun chain ->
    Array.fold_left (+.) 0. chain /. float_of_int chain_length
  ) chains in
  
  (* Overall mean *)
  let overall_mean = Array.fold_left (+.) 0. chain_means /. float_of_int n_chains in
  
  (* Between-chain variance *)
  let b = Array.fold_left (fun acc mean ->
    acc +. (mean -. overall_mean) ** 2.
  ) 0. chain_means in
  let b = b *. float_of_int chain_length /. float_of_int (n_chains - 1) in
  
  (* Within-chain variance *)
  let w = Array.fold_left (fun acc chain ->
    let chain_mean = Array.fold_left (+.) 0. chain /. float_of_int chain_length in
    let variance = Array.fold_left (fun acc x ->
      acc +. (x -. chain_mean) ** 2.
    ) 0. chain /. float_of_int (chain_length - 1) in
    acc +. variance
  ) 0. chains /. float_of_int n_chains in
  
  (* Potential scale reduction factor *)
  sqrt ((float_of_int (chain_length - 1) /. float_of_int chain_length *. w +. b /. float_of_int chain_length) /. w)

let assess_convergence samples chains =
  let p = size (List.hd samples).theta_star 0 in
  
  (* Convert samples to parameter-wise chains *)
  let param_chains = Array.init p (fun j ->
    Array.init (Array.length chains) (fun c ->
      Array.map (fun s -> get s.theta_star j) chains.(c)
    )
  ) in
  
  (* Compute diagnostics for each parameter *)
  Array.init p (fun j ->
    let merged_chain = Array.concat (Array.to_list param_chains.(j)) in
    {
      potential_scale_reduction = compute_psrf param_chains.(j);
      effective_sample_size = compute_ess merged_chain;
      autocorrelation = compute_acf merged_chain 20;
    }
  )