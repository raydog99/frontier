open Types
open Torch
open MatrixOps

let fisher_information model =
  let gamma = Tensor.zeros [model.spec.n_random; 1] in
  let (y_tilde, w) = Glmm.compute_working_response model gamma in
  let r = Glmm.compute_r model w (Tensor.zeros [2; 1]) in
  let r_inv = safe_inverse r in
  
  (* Compute blocks of the Fisher Information matrix *)
  let i_beta = Tensor.mm (Tensor.transpose model.x ~dim0:0 ~dim1:1)
                        (Tensor.mm r_inv model.x) in
  let i_omega = Tensor.mm (Tensor.transpose model.z ~dim0:0 ~dim1:1)
                         (Tensor.mm r_inv model.z) in
  
  (* Assemble full Fisher Information matrix *)
  Tensor.cat [
    Tensor.cat [i_beta; Tensor.zeros [model.spec.n_fixed; 2]] ~dim:1;
    Tensor.cat [Tensor.zeros [2; model.spec.n_fixed]; i_omega] ~dim:1
  ] ~dim:0

let compute_p_value stat df =
  let stat_float = Tensor.scalar_to_float stat |> Scalar.to_float in
  1.0 -. Stats.ChiSquared.cdf stat_float df

let likelihood_ratio_stat model1 model2 =
  let state1 = Algo.run model1 in
  let state2 = Algo.run model2 in
  
  let psi1 = Glmm.compute_psi model1 
              (Tensor.zeros [model1.spec.n_fixed; 1])
              (Tensor.zeros [2; 1])
              (Tensor.zeros [model1.spec.n_random; 1]) in
  let psi2 = Glmm.compute_psi model2
              (Tensor.zeros [model2.spec.n_fixed; 1])
              (Tensor.zeros [2; 1])
              (Tensor.zeros [model2.spec.n_random; 1]) in
  
  Tensor.sub psi2 psi1 |> Tensor.mul_scalar (Scalar.float 2.0)

let score_stat model1 model2 b =
  let b_star = moore_penrose_inverse b in
  let state1 = Algo.run model1 in
  
  let beta2_star = Tensor.mm b_star state1.beta in
  let omega2_star = Tensor.mm b_star state1.omega in
  let score = Glmm.gradient_alpha model2 beta2_star omega2_star 
                (Tensor.zeros [model2.spec.n_random; 1])
                model2.y in
  
  let info = fisher_information model2 in
  let info_inv = safe_inverse info in
  
  Tensor.mm (Tensor.transpose score ~dim0:0 ~dim1:1)
            (Tensor.mm info_inv score) |> Tensor.neg

let wald_stat model1 model2 b =
  let b_star = moore_penrose_inverse b in
  let state1 = Algo.run model1 in
  
  let beta2_star = Tensor.mm b_star state1.beta in
  let omega2_star = Tensor.mm b_star state1.omega in
  let params = Tensor.cat [beta2_star; omega2_star] ~dim:0 in
  
  let info = fisher_information model2 in
  Tensor.mm (Tensor.transpose params ~dim0:0 ~dim1:1)
            (Tensor.mm info params) |> Tensor.neg