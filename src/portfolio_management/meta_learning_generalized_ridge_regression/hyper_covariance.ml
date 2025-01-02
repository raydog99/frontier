open Torch
open Types
open Matrix_ops

let estimate_unregularized ~tasks ~sigma_sq =
  let dim = Tensor.size (List.hd tasks).x |> List.nth 1 in
  let n_tasks = List.length tasks in
  
  let compute_gradient omega =
    let grad = List.fold_left (fun acc task ->
      let xt = Tensor.transpose task.x ~dim0:0 ~dim1:1 in
      let yyt = Tensor.mm task.y (Tensor.transpose task.y ~dim0:0 ~dim1:1) in
      let p = float_of_int dim in
      let xox = Tensor.mm (Tensor.mm task.x omega) xt |> Tensor.div_scalar p in
      let diff = Tensor.sub yyt (Tensor.add xox (Tensor.eye dim |> Tensor.mul_scalar sigma_sq)) in
      let term = Tensor.mm (Tensor.mm xt diff) task.x |> Tensor.div_scalar p in
      Tensor.add acc term
    ) (Tensor.zeros [dim; dim]) tasks in
    Tensor.div_scalar grad (-4. *. float_of_int n_tasks)
  in
  
  (* Gradient descent with fixed step size *)
  let rec optimize omega step iter max_iter =
    if iter >= max_iter then omega
    else
      let grad = compute_gradient omega in
      let grad_norm = frobenius_norm grad in
      if grad_norm < 1e-6 then omega
      else
        let omega_new = Tensor.sub omega (Tensor.mul_scalar grad step) in
        let omega_new = nearest_positive_definite omega_new in
        optimize omega_new step (iter + 1) max_iter
  in
  
  optimize (Tensor.eye dim) 0.01 0 1000

let estimate_l1_regularized ~tasks ~sigma_sq ~lambda =
  let dim = Tensor.size (List.hd tasks).x |> List.nth 1 in
  let n_tasks = List.length tasks in
  
  let compute_smooth_gradient omega =
    let grad = List.fold_left (fun acc task ->
      let xt = Tensor.transpose task.x ~dim0:0 ~dim1:1 in
      let yyt = Tensor.mm task.y (Tensor.transpose task.y ~dim0:0 ~dim1:1) in
      let p = float_of_int dim in
      let xox = Tensor.mm (Tensor.mm task.x omega) xt |> Tensor.div_scalar p in
      let diff = Tensor.sub yyt (Tensor.add xox (Tensor.eye dim |> Tensor.mul_scalar sigma_sq)) in
      let term = Tensor.mm (Tensor.mm xt diff) task.x |> Tensor.div_scalar p in
      Tensor.add acc term
    ) (Tensor.zeros [dim; dim]) tasks in
    Tensor.div_scalar grad (-4. *. float_of_int n_tasks)
  in
  
  let soft_threshold x t =
    let sign = Tensor.sign x in
    let abs_x = Tensor.abs x in
    let threshold = Tensor.sub abs_x (Tensor.full_like abs_x t) in
    let threshold = Tensor.relu threshold in
    Tensor.mul sign threshold
  in
  
  let proximal_step omega step =
    let grad = compute_smooth_gradient omega in
    let omega_grad = Tensor.sub omega (Tensor.mul_scalar grad step) in
    
    (* Apply L1 proximal operator to off-diagonal elements *)
    let diag_mask = Tensor.eye dim in
    let off_diag = Tensor.mul (Tensor.sub (Tensor.ones_like diag_mask) diag_mask) omega_grad in
    let prox_off_diag = soft_threshold off_diag (lambda *. step) in
    let result = Tensor.add (Tensor.mul diag_mask omega_grad) prox_off_diag in
    nearest_positive_definite result
  in
  
  (* Main optimization loop *)
  let rec optimize omega step iter max_iter =
    if iter >= max_iter then omega
    else
      let omega_new = proximal_step omega step in
      let diff_norm = frobenius_norm (Tensor.sub omega_new omega) in
      if diff_norm < 1e-6 then omega_new
      else optimize omega_new step (iter + 1) max_iter
  in
  
  optimize (Tensor.eye dim) 0.01 0 1000

(* Correlation-based estimation *)
let estimate_correlation ~tasks ~l0 ~lambda =
  let dim = Tensor.size (List.hd tasks).x |> List.nth 1 in
  let full_rank_tasks = List.take l0 tasks in
  
  (* Estimate diagonal entries *)
  let w_hat = List.fold_left (fun acc task ->
    let xt = Tensor.transpose task.x ~dim0:0 ~dim1:1 in
    let xtx_inv = Tensor.inverse (Tensor.mm xt task.x) in
    let z = Tensor.mm (Tensor.mm xtx_inv xt) task.y in
    let zzt = Tensor.mm z (Tensor.transpose z ~dim0:0 ~dim1:1) in
    Tensor.add acc (Tensor.diag zzt)
  ) (Tensor.zeros [dim]) full_rank_tasks in
  
  let w_hat = Tensor.div_scalar w_hat (float_of_int l0) in
  let w_sqrt = Tensor.sqrt w_hat |> Tensor.diag in
  let w_inv_sqrt = Tensor.pow_scalar w_hat (-0.5) |> Tensor.diag in
  
  (* Estimate correlation matrix *)
  let remaining_tasks = List.drop l0 tasks in
  let theta_init = Tensor.eye dim in
  
  let estimate_theta tasks theta =
    estimate_l1_regularized ~tasks ~sigma_sq:0. ~lambda:lambda
  in
  
  let theta_hat = estimate_theta remaining_tasks theta_init in
  Tensor.mm (Tensor.mm w_sqrt theta_hat) w_sqrt