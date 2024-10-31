open Torch

let estimate_bounded_covariance ~samples ~epsilon =
  let n, d = Tensor.size samples 0, Tensor.size samples 1 in
  
  (* Initial conditions *)
  let nu = ref (Tensor.mean samples ~dim:[0]) in
  let max_iter = Int.of_float (log (Float.of_int d)) in
  
  for _ = 0 to max_iter - 1 do
    let matrices = Array.init n (fun i ->
      let sample = Tensor.slice samples ~dim:0 ~start:i ~end_:(i+1) in
      let centered = Tensor.sub sample !nu in
      Kronecker_ops.efficient_outer_product centered centered
    ) in
    
    let solution = Sdp_solver.solve ~matrices ~epsilon 
      ~max_iterations:1000 ~convergence_threshold:0.01 in
    
    if solution.objective <= 1.0 +. epsilon then
      nu := compute_weighted_mean samples solution.weights
    else
      let direction = compute_improvement_direction solution.dual_matrix in
      let step_size = compute_step_size direction samples epsilon in
      nu := update_estimate !nu direction step_size
  done;
  !nu

let estimate_known_covariance ~samples ~epsilon ~tau =
  let n, d = Tensor.size samples 0, Tensor.size samples 1 in
  
  (* Initial conditions *)
  let nu = ref (Tensor.mean samples ~dim:[0]) in
  let max_iter = Int.of_float (log (Float.of_int d)) in
  
  for _ = 0 to max_iter - 1 do
    let matrices = Array.init n (fun i ->
      let sample = Tensor.slice samples ~dim:0 ~start:i ~end_:(i+1) in
      let centered = Tensor.sub sample !nu in
      Kronecker_ops.efficient_outer_product centered centered
    ) in
    
    let solution = Sdp_solver.solve ~matrices ~epsilon:(epsilon *. tau)
      ~max_iterations:1000 ~convergence_threshold:(epsilon /. 10.0) in
    
    if solution.objective <= 1.0 +. tau then
      nu := compute_weighted_mean samples solution.weights
    else
      let direction = compute_improvement_direction solution.dual_matrix in
      let step_size = compute_optimal_step_size direction samples epsilon tau in
      nu := update_estimate !nu direction step_size
  done;
  !nu

let compute_weights ~samples ~center ~epsilon =
  let n = Tensor.size samples 0 in
  let matrices = Array.init n (fun i ->
    let sample = Tensor.slice samples ~dim:0 ~start:i ~end_:(i+1) in
    let centered = Tensor.sub sample center in
    Kronecker_ops.efficient_outer_product centered centered
  ) in
  
  let solution = Sdp_solver.solve ~matrices ~epsilon 
    ~max_iterations:1000 ~convergence_threshold:0.01 in
  solution.weights

let compute_weighted_mean samples weights =
  let weighted_samples = Tensor.mul 
    (Tensor.unsqueeze weights ~dim:1) samples in
  Tensor.sum weighted_samples ~dim:[0]

let compute_improvement_direction dual_matrix =
  let eigenvals, eigenvecs = 
    Numerical_stability.stable_eigendecomposition dual_matrix in
  let max_idx = Tensor.argmax eigenvals |> Tensor.int_value in
  Tensor.select eigenvecs ~dim:1 ~index:max_idx

let compute_step_size direction samples epsilon =
  let d = Tensor.size samples 1 in
  let projected = Tensor.mv samples direction in
  let median = compute_median projected in
  let mad = compute_median_absolute_deviation projected median in
  min (mad /. (epsilon *. sqrt (Float.of_int d))) 1.0

let update_estimate current direction step_size =
  Tensor.add current 
    (Tensor.mul_scalar direction step_size)