open Torch

type preconditioner = {
  matrix: Tensor.t;
  eigenvalues: Tensor.t;
  eigenvectors: Tensor.t;
  blocks: Tensor.t array;
  block_size: int;
  probabilities: float array;
}

type config = {
  dim: int;
  step_size: float;
  num_blocks: int;
  block_size: int;
  relative_tolerance: float;
  max_iterations: int;
  convergence_threshold: float;
}

type smoothness_params = {
  m: float;
  m_standard: float;
  big_m: float;
  big_m_standard: float;
}

type condition_numbers = {
  kappa: float;
  kappa_rel: float array;
  kappa_rel_total: float;
}

type block_statistics = {
  smoothness_violations: int;
  avg_gradient_norm: float;
  avg_step_size: float;
  success_rate: float;
  iterations: int;
}

type sampling_stats = {
  wasserstein_distances: float list;
  kl_divergences: float list;
  smoothness_violations: int;
  acceptance_rate: float;
  effective_sample_size: float;
}

type convergence_diagnostics = {
  has_converged: bool;
  error_estimate: float;
  theoretical_bound: float;
  actual_iterations: int;
  theoretical_iterations: int;
}

let check_pl_inequality ~f ~grad_f ~x ~x_star ~alpha =
  let grad_norm = Tensor.norm grad_f in
  let f_diff = Tensor.sub (f x) (f x_star) in
  Tensor.to_float0_exn f_diff <= 
    (1. /. (2. *. alpha)) *. (Tensor.to_float0_exn grad_norm) ** 2.

let wasserstein_distance x y =
  let diff = Tensor.sub x y in
  Tensor.norm diff |> Tensor.to_float0_exn

let kl_divergence x y =
  let ratio = Tensor.div x y in
  let log_ratio = Tensor.log ratio in
  Tensor.mean (Tensor.mul x log_ratio) |> Tensor.to_float0_exn

let compute_effective_sample_size samples =
  let n = List.length samples in
  let mean = List.fold_left (fun acc x -> 
    Tensor.add acc x) (List.hd samples) (List.tl samples) in
  let mean = Tensor.div_scalar mean (float_of_int n) in
  
  let auto_corr = List.mapi (fun i x ->
    let diff = Tensor.sub x mean in
    Tensor.sum (Tensor.mul diff diff) |> Tensor.to_float0_exn
  ) samples in
  
  let sum_auto_corr = List.fold_left (+.) 0. auto_corr in
  float_of_int n /. (1. +. 2. *. sum_auto_corr)

let eigendecomposition matrix =
  Tensor.symeig matrix ~eigenvectors:true

let create_blocks matrix block_size =
  let dim = Tensor.size matrix |> List.hd in
  let num_blocks = dim / block_size in
  Array.init num_blocks (fun i ->
    let start_idx = i * block_size in
    Tensor.narrow matrix ~dim:0 ~start:start_idx ~length:block_size
  )

let compute_block_gram block =
  let block_t = Tensor.transpose block ~dim0:0 ~dim1:1 in
  Tensor.matmul block block_t

let project_to_block tensor block =
  let block_t = Tensor.transpose block ~dim0:0 ~dim1:1 in
  Tensor.matmul block (Tensor.matmul block_t tensor)

let compute_condition_number matrix =
  let eigenvals, _ = eigendecomposition matrix in
  let max_eval = Tensor.max eigenvals |> Tensor.to_float0_exn in
  let min_eval = Tensor.min eigenvals |> Tensor.to_float0_exn in
  max_eval /. min_eval

let compute_relative_condition matrix params =
  let cond = compute_condition_number matrix in
  cond *. params.m /. params.big_m

let create matrix block_size params =
  let eigenvalues, eigenvectors = eigendecomposition matrix in
  let blocks = create_blocks eigenvectors block_size in
  let initial_probs = Array.make (Array.length blocks) 
    (1. /. float_of_int (Array.length blocks)) in
  {
    matrix;
    eigenvalues;
    eigenvectors;
    blocks;
    block_size;
    probabilities = initial_probs;
  }

let update precond block_stats =
  let new_probs = Array.mapi (fun i stats ->
    let success_weight = stats.success_rate in
    let grad_weight = 1. /. (1. +. stats.avg_gradient_norm) in
    success_weight *. grad_weight
  ) block_stats in
  let sum = Array.fold_left (+.) 0. new_probs in
  let normalized_probs = Array.map (fun p -> p /. sum) new_probs in
  { precond with probabilities = normalized_probs }

let sample_block precond block_stats =
  let cumsum = Array.make (Array.length precond.probabilities) 0. in
  Array.iteri (fun i p ->
    cumsum.(i) <- p +. (if i > 0 then cumsum.(i-1) else 0.)
  ) precond.probabilities;
  
  let r = Random.float 1.0 in
  let idx = ref 0 in
  while !idx < Array.length cumsum && r > cumsum.(!idx) do
    incr idx
  done;
  (!idx, precond.blocks.(!idx))

let compute_optimal_probabilities precond params =
  Array.mapi (fun i block ->
    let block_cond = compute_relative_condition 
      (compute_block_gram block) params in
    1. /. (1. +. block_cond)
  ) precond.blocks

let compute_optimal_step_size ~grad_potential ~params ~x =
  let grad = grad_potential x in
  let grad_norm = Tensor.norm grad |> Tensor.to_float0_exn in
  min (1. /. grad_norm) (params.m /. (4. *. params.big_m))

let lmc ~potential ~grad_potential ~config ~x =
  let grad = grad_potential x in
  let step = Tensor.mul_scalar grad (-.config.step_size) in
  let noise = Tensor.randn (Tensor.size x) in
  let noise_term = Tensor.mul_scalar noise 
    (Float.sqrt (2. *. config.step_size)) in
  let next_x = Tensor.add (Tensor.add x step) noise_term in
  
  let dist = wasserstein_distance x next_x in
  let kl = kl_divergence x next_x in
  let stats = {
    wasserstein_distances = [dist];
    kl_divergences = [kl];
    smoothness_violations = 0;
    acceptance_rate = 1.0;
    effective_sample_size = 1.0;
  } in
  (next_x, stats)

let plmc ~potential ~grad_potential ~config ~precond ~x =
  let grad = grad_potential x in
  let precond_grad = Tensor.matmul precond.matrix grad in
  let step = Tensor.mul_scalar precond_grad (-.config.step_size) in
  let noise = Tensor.randn (Tensor.size x) in
  let noise_term = 
    let scaled_noise = Tensor.matmul 
      (Tensor.matrix_power precond.matrix 0.5) noise in
    Tensor.mul_scalar scaled_noise (Float.sqrt (2. *. config.step_size))
  in
  let next_x = Tensor.add (Tensor.add x step) noise_term in
  
  let dist = wasserstein_distance x next_x in
  let kl = kl_divergence x next_x in
  let stats = {
    wasserstein_distances = [dist];
    kl_divergences = [kl];
    smoothness_violations = 0;
    acceptance_rate = 1.0;
    effective_sample_size = 1.0;
  } in
  (next_x, stats)

let init ~config ~precond ~params ~x0 =
  Array.make (Array.length precond.blocks) {
    smoothness_violations = 0;
    avg_gradient_norm = 0.;
    avg_step_size = config.step_size;
    success_rate = 1.0;
    iterations = 0;
  }

let step ~potential ~grad_potential ~config ~precond ~params ~block_stats ~x =
  let block_idx, block = sample_block precond block_stats in
  let grad = grad_potential x in
  let proj_grad = project_to_block grad block in
  
  let step_size = compute_optimal_step_size 
    ~grad_potential ~params ~x in
  let step = Tensor.mul_scalar proj_grad (-.step_size) in
  
  let noise = Tensor.randn (Tensor.size step) in
  let noise_term = Tensor.mul_scalar noise 
    (Float.sqrt (2. *. step_size)) in
  let next_x = Tensor.add (Tensor.add x step) noise_term in
  
  let success = verify_smoothness_conditions 
    ~grad_potential ~x ~y:next_x ~params in
  let grad_norm = Tensor.norm proj_grad |> Tensor.to_float0_exn in
  
  let new_stats = update_block_statistics 
    ~stats:block_stats
    ~block_idx
    ~success
    ~grad_norm
    ~step_size in
  
  let sampling_stats = {
    wasserstein_distances = [wasserstein_distance x next_x];
    kl_divergences = [kl_divergence x next_x];
    smoothness_violations = if success then 0 else 1;
    acceptance_rate = if success then 1.0 else 0.0;
    effective_sample_size = 1.0;
  } in
  
  (next_x, new_stats, sampling_stats)

let run_chain ~potential ~grad_potential ~config ~precond ~params ~x0 =
  let block_stats = init ~config ~precond ~params ~x0 in
  let rec loop x stats samples iter =
    if iter >= config.max_iterations then
      let diagnostics = check_convergence 
        ~samples ~config ~params in
      (List.rev samples, diagnostics)
    else
      let next_x, new_stats, _ = step 
        ~potential ~grad_potential ~config ~precond ~params 
        ~block_stats:stats ~x in
      loop next_x new_stats (next_x :: samples) (iter + 1)
  in
  loop x0 block_stats [x0] 0

let verify_smoothness_conditions ~grad_potential ~x ~y ~params =
  let grad_x = grad_potential x in
  let grad_y = grad_potential y in
  let diff_grads = Tensor.sub grad_x grad_y in
  let diff_points = Tensor.sub x y in
  let lhs = Tensor.norm diff_grads |> Tensor.to_float0_exn in
  let rhs = params.big_m *. (Tensor.norm diff_points |> Tensor.to_float0_exn) in
  lhs <= rhs

let compute_complexity_bound ~config ~params ~epsilon =
  let r = float_of_int config.block_size in
  let d = float_of_int config.dim in
  let kappa = params.big_m /. params.m in
  let log_term = log (1. /. epsilon) in
  let iterations = ceil (d *. d *. kappa *. log_term /. 
    (epsilon *. epsilon *. r)) in
  int_of_float iterations

let estimate_mixing_time ~config ~params ~initial_dist ~target_error =
  let kappa = params.big_m /. params.m in
  let d = float_of_int config.dim in
  let mixing_time = ceil (kappa *. d *. 
    log (initial_dist /. target_error)) in
  int_of_float mixing_time

let verify_convergence_rate ~samples ~params =
  let dists = List.map2 (fun x y ->
    wasserstein_distance x y
  ) (List.tl samples) samples in
  
  let rates = List.map2 (fun d1 d2 -> d2 /. d1) 
    (List.tl dists) dists in
  let avg_rate = List.fold_left (+.) 0. rates /. 
    float_of_int (List.length rates) in
  let expected_rate = exp (-. params.m /. (4. *. params.big_m)) in
  (avg_rate, avg_rate <= expected_rate)

let adapt_step_size ~current ~success_rate ~smoothness =
  if success_rate > 0.7 then
    min (current *. 1.1) (1. /. smoothness)
  else if success_rate < 0.3 then
    current *. 0.9
  else
    current

let adapt_block_size ~config ~stats ~condition_number =
  let avg_success_rate = Array.fold_left (fun acc s -> 
    acc +. s.success_rate) 0. stats /. 
    float_of_int (Array.length stats) in
  
  let current_size = config.block_size in
  if avg_success_rate < 0.3 && current_size > 1 then
    max 1 (current_size / 2)
  else if avg_success_rate > 0.7 && 
          current_size < config.dim && 
          condition_number < 100. then
    min config.dim (current_size * 2)
  else
    current_size

let adapt_probabilities ~precond ~stats =
  Array.mapi (fun i stat ->
    let efficiency = stat.success_rate /. 
      (1. +. stat.avg_gradient_norm) in
    efficiency
  ) stats |> fun probs ->
  let sum = Array.fold_left (+.) 0. probs in
  Array.map (fun p -> p /. sum) probs

let update_block_statistics ~stats ~block_idx ~success ~grad_norm ~step_size =
  let new_stats = Array.copy stats in
  let current = stats.(block_idx) in
  new_stats.(block_idx) <- {
    smoothness_violations = 
      current.smoothness_violations + if success then 0 else 1;
    avg_gradient_norm = 
      0.9 *. current.avg_gradient_norm +. 0.1 *. grad_norm;
    avg_step_size = 
      0.9 *. current.avg_step_size +. 0.1 *. step_size;
    success_rate = 
      0.9 *. current.success_rate +. 0.1 *. (if success then 1. else 0.);
    iterations = current.iterations + 1;
  };
  new_stats

let check_convergence ~samples ~config ~params =
  let error = compute_effective_sample_size samples in
  let theoretical = compute_complexity_bound 
    ~config ~params ~epsilon:config.convergence_threshold in
  let actual = List.length samples in
  let rate, rate_ok = verify_convergence_rate 
    ~samples ~params in
  
  {
    has_converged = error <= config.convergence_threshold && rate_ok;
    error_estimate = error;
    theoretical_bound = float_of_int theoretical;
    actual_iterations = actual;
    theoretical_iterations = theoretical;
  }

let compute_error_estimate ~samples ~target =
  match target with
  | Some t -> 
      List.map (fun s -> wasserstein_distance s t) samples
      |> fun dists -> List.fold_left (+.) 0. dists /. 
                     float_of_int (List.length dists)
  | None ->
      let n = List.length samples in
      List.mapi (fun i s ->
        List.mapi (fun j s' ->
          if i < j then wasserstein_distance s s'
          else 0.
        ) samples |> List.fold_left (+.) 0.
      ) samples |> fun dists ->
      List.fold_left (+.) 0. dists /. (float_of_int (n * (n - 1) / 2))

let monitor_chain ~samples ~stats =
  let n = List.length samples in
  Printf.printf "Chain status after %d iterations:\n" n;
  Printf.printf "  Average W2 distance: %.6f\n" 
    (List.fold_left (+.) 0. stats.wasserstein_distances /. 
     float_of_int (List.length stats.wasserstein_distances));
  Printf.printf "  Acceptance rate: %.2f%%\n" 
    (100. *. stats.acceptance_rate);
  Printf.printf "  Effective sample size: %.2f\n" 
    stats.effective_sample_size;
  Printf.printf "  Smoothness violations: %d\n" 
    stats.smoothness_violations

let generate_diagnostics_report ~samples ~stats ~diagnostics =
  let buffer = Buffer.create 1024 in
  Buffer.add_string buffer "SLMC Sampling Report\n";
  Buffer.add_string buffer "==================\n\n";
  
  Buffer.add_string buffer (Printf.sprintf 
    "Convergence status: %s\n" 
    (if diagnostics.has_converged then "Converged" else "Not converged"));
  
  Buffer.add_string buffer (Printf.sprintf 
    "Error estimate: %.6f (threshold: %.6f)\n"
    diagnostics.error_estimate diagnostics.theoretical_bound);
  
  Buffer.add_string buffer (Printf.sprintf 
    "Iterations: %d (theoretical: %d)\n"
    diagnostics.actual_iterations diagnostics.theoretical_iterations);
  
  Buffer.add_string buffer (Printf.sprintf 
    "Effective sample size: %.2f\n" stats.effective_sample_size);
  
  Buffer.add_string buffer (Printf.sprintf 
    "Final acceptance rate: %.2f%%\n" (100. *. stats.acceptance_rate));
  
  Buffer.contents buffer