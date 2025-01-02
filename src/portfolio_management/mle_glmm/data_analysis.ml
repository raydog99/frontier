open Types
open Torch
open MatrixOps

let model_comparison models =
  List.map (fun model -> 
    let state = Algo.run model in
    let (aic, _) = Inference.information_criteria model state in
    (model, aic)
  ) models

let fixed_effects_summary model =
  let state = Algo.run model in
  let se = Inference.standard_errors model state in
  let z_stats = Tensor.div state.beta se in
  
  let p_values = Tensor.map z_stats ~f:(fun z ->
    2.0 *. (1.0 -. Stats.Normal.cdf (Float.abs z) 0.0 1.0)) in
  
  (state.beta, se, z_stats, p_values)

let residual_analysis model state =
  let fitted = Tensor.add (Tensor.mm model.x state.beta)
                         (Tensor.mm model.z state.gamma) in
  
  (* Pearson residuals *)
  let pearson = match model.spec.distribution with
    | Binomial {trials} ->
        let p = Tensor.sigmoid fitted in
        let var = Tensor.mul (Tensor.mul trials p) 
                            (Tensor.sub (Tensor.ones_like p) p) in
        Tensor.div (Tensor.sub model.y fitted) (Tensor.sqrt var)
    | Poisson ->
        let mu = Tensor.exp fitted in
        Tensor.div (Tensor.sub model.y mu) (Tensor.sqrt mu)
    | Normal {variance} ->
        Tensor.div (Tensor.sub model.y fitted) 
                  (Tensor.scalar_float (sqrt variance)) in
  
  (* Deviance residuals *)
  let deviance = match model.spec.distribution with
    | Binomial {trials} ->
        let y_n = Tensor.div model.y trials in
        let mu_n = Tensor.sigmoid fitted in
        let term1 = Tensor.mul y_n (Tensor.log (Tensor.div y_n mu_n)) in
        let term2 = Tensor.mul (Tensor.sub (Tensor.ones_like y_n) y_n)
                              (Tensor.log (Tensor.div 
                                (Tensor.sub (Tensor.ones_like y_n) y_n)
                                (Tensor.sub (Tensor.ones_like mu_n) mu_n))) in
        Tensor.sqrt (Tensor.mul_scalar (Tensor.add term1 term2)
                                     (Scalar.float 2.0))
    | Poisson ->
        let mu = Tensor.exp fitted in
        let term1 = Tensor.mul model.y (Tensor.log (Tensor.div model.y mu)) in
        let term2 = Tensor.sub model.y mu in
        Tensor.sqrt (Tensor.mul_scalar (Tensor.sub term1 term2)
                                     (Scalar.float 2.0))
    | Normal {variance} ->
        pearson in
  
  (* Leverage (hat matrix diagonal) *)
  let x_aug = Tensor.cat [model.x; model.z] ~dim:1 in
  let h_mat = Tensor.mm x_aug 
                (Tensor.mm (safe_inverse 
                            (Tensor.mm (Tensor.transpose x_aug ~dim0:0 ~dim1:1)
                                     x_aug))
                          (Tensor.transpose x_aug ~dim0:0 ~dim1:1)) in
  let leverage = Tensor.diagonal h_mat ~dim1:0 ~dim2:1 in
  
  (pearson, deviance, leverage)

let influence_measures model state =
  let (pearson, _, leverage) = residual_analysis model state in
  let scaled_residuals = Tensor.div (Tensor.pow pearson (Tensor.scalar_float 2.0))
                                  (Tensor.sub (Tensor.ones_like leverage) leverage) in
  let cooks_d = Tensor.mul scaled_residuals 
                          (Tensor.div leverage 
                                     (Tensor.scalar_float 
                                        (float model.spec.n_fixed))) in
  cooks_d