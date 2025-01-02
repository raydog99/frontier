open Torch

let create_default_config () = {
  Types.
  epsilon = 0.1;
  batch_size = 1000;
  max_iterations = 100;
  memory_limit = 1024 * 1024 * 1024; (* 1GB *)
  estimation_type = `Multiplicative;
}

let estimate ?(config = create_default_config()) ~samples () =
  match config.estimation_type with
  | `Multiplicative -> 
      estimate_multiplicative ~samples ~epsilon:config.epsilon
  | `Additive -> 
      estimate_additive ~samples ~epsilon:config.epsilon

let estimate_multiplicative ~samples ~epsilon =
  let n, d = Tensor.size samples 0, Tensor.size samples 1 in
  
  let required_samples = d * d * Int.of_float (log (1.0 /. epsilon)) in
  if n < required_samples then
    raise (Invalid_argument "Insufficient samples");
  
  let sigma_0 = Covariance_bounds.compute_initial_bound ~samples ~epsilon in
  
  (* First phase: Get to O(sqrt(epsilon)) error *)
  let t1 = Int.of_float (log (Float.of_int d)) in
  let sigma_t1 = ref sigma_0 in
  let history = ref [sigma_0] in
  
  for _ = 0 to t1 - 1 do
    let next_estimate = Covariance_refiner.refine_sqrt_epsilon
      ~current:!sigma_t1 ~samples ~epsilon in
    history := next_estimate :: !history;
    sigma_t1 := next_estimate
  done;
  
  (* Second phase: Get to O(epsilon log(1/epsilon)) error *)
  let t2 = t1 + Int.of_float (log (log (1.0 /. epsilon))) in
  let tau = ref (sqrt epsilon) in
  let sigma_t2 = ref !sigma_t1 in
  
  for _ = t1 to t2 - 1 do
    tau := sqrt !tau *. epsilon *. log (1.0 /. epsilon);
    let next_estimate = Covariance_refiner.refine_log_epsilon
      ~current:!sigma_t2 ~samples ~epsilon ~tau:!tau in
    history := next_estimate :: !history;
    sigma_t2 := next_estimate
  done;
  
  (* Compute error bounds *)
  let error_bounds = Error_tracker.compute_bounds
    ~estimate:!sigma_t2
    ~true_cov:(compute_empirical_covariance samples) in
  
  (!sigma_t2, error_bounds)

let estimate_additive ~samples ~epsilon =
  let m0 = Algorithm4.compute_crude_estimate samples epsilon in
  
  (* Partition into subspaces *)
  let partition = Subspace_decomposition.decompose ~matrix:m0 ~epsilon in
  
  (* Process each subspace *)
  let s1_samples = Subspace_decomposition.project_samples 
    ~samples ~subspace:partition.high_eigenspace in
  let m1 = Covariance_refiner.refine_sqrt_epsilon 
    ~current:m0 ~samples:s1_samples ~epsilon in
  
  let s12_samples = Subspace_decomposition.project_samples 
    ~samples ~subspace:(Tensor.cat 
      [partition.high_eigenspace; partition.medium_eigenspace] ~dim:1) in
  let m2 = estimate_multiplicative ~samples:s12_samples ~epsilon |> fst in
  
  let scaled_samples = scale_samples samples partition epsilon in
  let m3 = Algorithm4.compute_crude_estimate scaled_samples epsilon in
  
  (* Combine estimates *)
  let final_estimate = Subspace_decomposition.combine_estimates
    ~high:m1 ~medium:m2 ~cross_terms:m3 ~partition ~epsilon in
  
  let error_bounds = Error_tracker.compute_bounds
    ~estimate:final_estimate
    ~true_cov:(compute_empirical_covariance samples) in
  
  (final_estimate, error_bounds)

let verify_estimate ~estimate ~samples ~epsilon ~estimation_type =
  let true_cov = compute_empirical_covariance samples in
  let error_bounds = Error_tracker.compute_bounds ~estimate ~true_cov in
  
  let valid = match estimation_type with
  | `Multiplicative -> 
      error_bounds.multiplicative <= epsilon *. log (1.0 /. epsilon)
  | `Additive ->
      error_bounds.relative <= epsilon in
  
  (valid, error_bounds)

let compute_empirical_covariance samples =
  let n = Tensor.size samples 0 in
  let centered = center_samples samples in
  let cov = Tensor.mm 
    (Tensor.transpose centered ~dim0:0 ~dim1:1) centered in
  Tensor.div_scalar cov (Float.of_int n)

let scale_samples samples partition epsilon =
  let scale_factors = compute_scale_factors partition epsilon in
  Tensor.mul samples 
    (Tensor.unsqueeze scale_factors ~dim:0)