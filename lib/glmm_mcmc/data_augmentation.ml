open Torch
open Types

module Probit = struct
  type augmented_state = {
    params: model_params;
    v: Tensor.t;
  }

  let sample_latent data params =
    let eta = Glmm_base.linear_predictor data.x data.z params.beta params.u in
    let v = Tensor.zeros (Tensor.size data.y) in
    
    (* Sample from truncated normal *)
    for i = 0 to Tensor.size data.y 0 - 1 do
      let yi = Tensor.float_value (Tensor.get data.y i) in
      let mu = Tensor.float_value (Tensor.get eta i) in
      let sample =
        if yi > 0.5 then
          (* Sample from N(mu,1) truncated to (0,inf) *)
          let rec sample_pos () =
            let z = Tensor.float_value (Tensor.randn [1]) +. mu in
            if z > 0.0 then z else sample_pos ()
          in
          sample_pos ()
        else
          (* Sample from N(mu,1) truncated to (-inf,0) *)
          let rec sample_neg () =
            let z = Tensor.float_value (Tensor.randn [1]) +. mu in
            if z <= 0.0 then z else sample_neg ()
          in
          sample_neg ()
      in
      Tensor.set v i (Tensor.float_tensor [sample])
    done;
    v

  let step data state prior =
    (* Sample latent variables *)
    let v = sample_latent data state.params in
    
    (* Sample beta *)
    let precision = Tensor.(
      add
        (mm (transpose data.x) data.x)
        prior.q
    ) in
    let mean = Tensor.(mm (inverse precision)
      (add
        (mm (transpose data.x) (sub v (mm data.z state.params.u)))
        (mm prior.q prior.mu_0))) in
    let beta = Glmm_base.rmvnorm mean (Numerical.safe_inverse precision) in
    
    (* Sample u *)
    let precision_u = Tensor.(
      add
        (mm (transpose data.z) data.z)
        (mul state.params.lambda (eye (size state.params.u 0)))
    ) in
    let mean_u = Tensor.(mm (inverse precision_u)
      (mm (transpose data.z) (sub v (mm data.x beta)))) in
    let u = Glmm_base.rmvnorm mean_u (Numerical.safe_inverse precision_u) in
    
    (* Sample lambda *)
    let shape = Tensor.(add prior.a (float_tensor [Float.of_int (size u 0) /. 2.0])) in
    let rate = Tensor.(add prior.b (div (sum (mul u u)) (float_tensor [2.0]))) in
    let lambda = Glmm_base.rgamma shape rate in
    
    {params = {beta; u; lambda}; v}
end

module Polya_Gamma = struct
  let sample n c =
    (* Approximation using sum of weighted gamma random variables *)
    let k_max = 100 in  (* Truncation level *)
    let sum = ref 0.0 in
    for k = 0 to k_max - 1 do
      let d = Float.of_int (2 * k + 1) in
      let shape = n /. 2.0 in
      let rate = d *. d *. Float.pi *. Float.pi /. 8.0 +. c *. c /. 2.0 in
      sum := !sum +. Tensor.float_value (Glmm_base.rgamma 
        (Tensor.float_tensor [shape])
        (Tensor.float_tensor [rate]))
    done;
    !sum *. 2.0
end