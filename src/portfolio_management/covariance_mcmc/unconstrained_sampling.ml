open Torch
open Types

type proximal_config = {
  step_size: float;
  n_steps: int;
  beta: float;
  epsilon: float;
}

let proximal_step gradient x y h =
  let grad_y = gradient y in
  let diff = Tensor.sub x y in
  let scaled_diff = Tensor.div_scalar diff h in
  Tensor.sub x (Tensor.mul_scalar (Tensor.add grad_y scaled_diff) h)

let estimate_covariance_with_guarantee gradient dimension beta epsilon =
  let h = 1.0 /. (2.0 *. beta) in
  let n0 = int_of_float (beta *. float_of_int dimension *. log (float_of_int dimension)) in
  let k = int_of_float (float_of_int dimension /. (epsilon *. epsilon)) in
  
  let chain = {
    kernel = (fun x ->
      let y = Tensor.add x 
        (Tensor.mul_scalar (Tensor.randn [dimension]) (sqrt h)) in
      proximal_step gradient x y h
    );
    stationary_dist = (fun x ->
      let norm = Tensor.norm x ~p:(Scalar 2) ~dim:[0] ~keepdim:false in
      exp (-. beta *. (Tensor.item norm) ** 2.0 /. 2.0)
    );
  } in
  
  (* Generate samples with burn-in *)
  let x0 = Tensor.div_scalar (Tensor.randn [dimension]) (sqrt beta) in
  let _ = Sampling.generate_chain_samples chain x0 n0 in
  let samples = Sampling.generate_chain_samples chain x0 k in
  
  let dist = {
    mean = Tensor.mean samples ~dim:[0] ~keepdim:false;
    covariance = Tensor.mm (Tensor.transpose samples 0 1) samples 
                 |> Tensor.div_scalar (float_of_int k);
  } in
  
  (* Verify guarantee *)
  let true_cov = Tensor.div_scalar (Tensor.eye dimension) beta in
  let diff = Tensor.sub dist.covariance true_cov in
  let guarantee_met = 
    Tensor.norm diff ~p:(Scalar 2) |> Tensor.item <= 
    epsilon *. (Tensor.norm true_cov ~p:(Scalar 2) |> Tensor.item) in
  
  dist, guarantee_met

let estimate_high_dim gradient dimension config device_config =
  let batch_size = 1000 in
  let total_samples = config.n_steps in
  
  let chain = {
    kernel = (fun x ->
      let batch = Tensor.randn [batch_size; dimension] in
      let proposals = Tensor.add 
        (Tensor.expand x [batch_size; dimension])
        (Tensor.mul_scalar batch (sqrt config.step_size)) in
      let results = Tensor.stack 
        (List.init batch_size (fun i ->
          proximal_step gradient x 
            (Tensor.select proposals 0 i) 
            config.step_size)) ~dim:0 in
      Tensor.mean results ~dim:[0] ~keepdim:false
    );
    stationary_dist = (fun x ->
      let norm = Tensor.norm x ~p:(Scalar 2) ~dim:[0] ~keepdim:false in
      exp (-. config.beta *. (Tensor.item norm) ** 2.0 /. 2.0)
    );
  } in
  
  let samples = Parallel_processing.parallel_chain_execution
    chain
    (Tensor.div_scalar (Tensor.randn [dimension]) (sqrt config.beta))
    8  (* number of parallel chains *)
    (total_samples / 8)
    device_config in
  
  let all_samples = Tensor.cat (Array.of_list samples) ~dim:0 in
  let conv_stats = Convergence_diagnostics.analyze_convergence 
    samples (total_samples / 80) in
  
  let dist = {
    mean = Tensor.mean all_samples ~dim:[0] ~keepdim:false;
    covariance = Tensor.mm (Tensor.transpose all_samples 0 1) all_samples
                 |> Tensor.div_scalar (float_of_int total_samples);
  } in
  
  dist, conv_stats