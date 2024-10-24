open Torch
open Types

type rounding_config = {
  radius: float;
  step_size: float;
  n_iterations: int;
  batch_size: int;
  tolerance: float;
}

type rounding_state = {
  transform: Tensor.t;
  radius: float;
  samples: Tensor.t;
  iteration: int;
}

let check_isotropic cov tolerance =
  let d = Tensor.size cov 0 in
  let eigenvals = Tensor.linalg_eigvals cov in
  let max_eig = Tensor.max eigenvals |> fst |> Tensor.item in
  let min_eig = Tensor.min eigenvals |> fst |> Tensor.item in
  abs_float (max_eig -. 1.0) <= tolerance && 
  abs_float (min_eig -. 1.0) <= tolerance

let generate_samples membership_oracle config x0 =
  let d = Tensor.size x0 0 in
  let h = config.step_size in
  
  let rec iterate x remaining acc =
    if remaining <= 0 then Tensor.cat (List.rev acc) ~dim:0
    else
      let y = Tensor.add x (Tensor.mul_scalar (Tensor.randn [d]) (sqrt h)) in
      if membership_oracle y then
        let next_x = Tensor.add y (Tensor.mul_scalar (Tensor.randn [d]) (sqrt h)) in
        if membership_oracle next_x then
          iterate next_x (remaining - 1) (next_x :: acc)
        else
          iterate x remaining acc
      else
        iterate x remaining acc
  in
  
  iterate x0 config.batch_size []

let isotropize membership_oracle dimension initial_transform config =
  let rec rounding_iteration state =
    if state.radius *. state.radius >= float_of_int dimension /. 10.0 then
      let mean = Tensor.mean state.samples ~dim:[0] ~keepdim:false in
      let centered = Tensor.sub state.samples (Tensor.expand_as mean state.samples) in
      let cov = Tensor.mm (Tensor.transpose centered 0 1) centered in
      let cov = Tensor.div_scalar cov (float_of_int (Tensor.size state.samples 0)) in
      { mean; covariance = cov }
    else
      (* Generate new samples *)
      let new_samples = generate_samples membership_oracle config (Tensor.zeros [dimension]) in
      
      (* Update transform *)
      let cov = Tensor.mm (Tensor.transpose new_samples 0 1) new_samples in
      let cov = Tensor.div_scalar cov (float_of_int (Tensor.size new_samples 0)) in
      let u, s, vh = Tensor.linalg_svd cov ~some:false in
      let mask = Tensor.le s (Tensor.full_like s (float_of_int dimension)) in
      let proj = Tensor.zeros_like mask in
      Tensor.copy_ proj mask;
      let proj_mat = Tensor.mm u (Tensor.mm (Tensor.diag proj) vh) in
      let m = Tensor.add (Tensor.eye dimension) proj_mat in
      let new_transform = Tensor.mm m state.transform in
      
      rounding_iteration {
        transform = new_transform;
        radius = 2.0 *. state.radius *. (1.0 -. 1.0 /. log (float_of_int dimension));
        samples = new_samples;
        iteration = state.iteration + 1;
      }
  in
  
  let initial_state = {
    transform = initial_transform;
    radius = 0.25;
    samples = Tensor.zeros [1; dimension];
    iteration = 0;
  } in
  
  rounding_iteration initial_state

let isotropize_memory_efficient membership_oracle dimension initial_transform batch_config =
  let batch_size = ref (Batch_processing.optimal_batch_size dimension batch_config) in
  
  let config = {
    radius = 0.25;
    step_size = 1.0 /. (10.0 *. float_of_int dimension ** 2.0);
    n_iterations = !batch_size;
    batch_size = !batch_size;
    tolerance = 0.1;
  } in
  
  let convergence_history = ref [] in
  
  let sample_generator () =
    let samples = generate_samples membership_oracle config (Tensor.zeros [dimension]) in
    
    (* Track convergence *)
    let cov = Tensor.mm (Tensor.transpose samples 0 1) samples in
    let cov = Tensor.div_scalar cov (float_of_int (Tensor.size samples 0)) in
    let conv_measure = check_isotropic cov config.tolerance in
    convergence_history := conv_measure :: !convergence_history;
    
    (* Adapt batch size *)
    batch_size := Batch_processing.optimal_batch_size 
      dimension 
      {batch_config with max_batch_size = !batch_size * 2};
    
    Some samples
  in
  
  let mean, cov = Batch_processing.streaming_covariance sample_generator dimension in
  { mean; covariance = cov }, Array.of_list !convergence_history