open Torch
open Types

let compute_expected data params =
  let eta = Glmm_base.linear_predictor data.x data.z params.beta params.u in
  let mu = Models.Logistic.link eta in
  let w = Tensor.(mul mu (sub (float_tensor [1.0]) mu)) in
  let w_mat = Tensor.diag w in
  
  (* Construct full Fisher information matrix *)
  let x_block = Tensor.(mm (mm (transpose data.x) w_mat) data.x) in
  let z_block = Tensor.(mm (mm (transpose data.z) w_mat) data.z) in
  let xz_block = Tensor.(mm (mm (transpose data.x) w_mat) data.z) in
  
  (* Add prior information *)
  let lambda_block = Tensor.(
    div
      (float_tensor [1.0])
      (mul (float_tensor [2.0]) (mul params.lambda params.lambda))
  ) in
  
  (* Construct block matrix *)
  Tensor.cat [
    Tensor.cat [x_block; xz_block] 1;
    Tensor.cat [Tensor.transpose xz_block 0 1; 
                Tensor.add z_block 
                  (Tensor.mul params.lambda (Tensor.eye (Tensor.size z_block 0)))] 1;
  ] 0

let observed_information data params =
  let eta = Glmm_base.linear_predictor data.x data.z params.beta params.u in
  let mu = Models.Logistic.link eta in
  let score = Tensor.(sub data.y mu) in
  
  (* Compute second derivatives *)
  let d2_beta = Tensor.(
    add
      (mm (mm (transpose data.x) (diag (neg mu))) data.x)
      (mm (mm (transpose data.x) (diag score)) data.x)
  ) in
  
  let d2_u = Tensor.(
    add
      (mm (mm (transpose data.z) (diag (neg mu))) data.z)
      (mm (mm (transpose data.z) (diag score)) data.z)
  ) in
  
  let d2_mixed = Tensor.(
    add
      (mm (mm (transpose data.x) (diag (neg mu))) data.z)
      (mm (mm (transpose data.x) (diag score)) data.z)
  ) in
  
  (* Construct observed information matrix *)
  Tensor.cat [
    Tensor.cat [d2_beta; d2_mixed] 1;
    Tensor.cat [Tensor.transpose d2_mixed 0 1; d2_u] 1;
  ] 0

let compute data params =
  let expected = compute_expected data params in
  let observed = observed_information data params in
  Tensor.(div (add expected observed) (float_tensor [2.0]))