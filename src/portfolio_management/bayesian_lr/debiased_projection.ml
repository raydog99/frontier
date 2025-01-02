open Torch
open Type
open LinAlg

let compute_residuals x j lambda_x =
  let p = size x 1 in
  let x_j = select x 1 j in
  let x_minus_j = cat [
    narrow x 1 0 j;
    narrow x 1 (j+1) (p-j-1)
  ] 1 in
  
  (* Solve LASSO for x_j on x_minus_j *)
  let beta = ref (zeros [p-1]) in
  let max_iter = 1000 in
  let tol = 1e-6 in
  
  let rec optimize iter beta_prev =
    if iter >= max_iter then beta_prev
    else
      let pred = matmul x_minus_j beta_prev in
      let grad = matmul (transpose2D x_minus_j ~dim0:0 ~dim1:1) (sub pred x_j) in
      let beta_next = sub beta_prev (scalar_mul 0.01 grad) in
      
      (* Soft thresholding *)
      let beta_thresh = map (fun x -> 
        let abs_x = abs_float x in
        if abs_x <= lambda_x then 0.
        else copysign (abs_x -. lambda_x) x
      ) beta_next in
      
      if norm (sub beta_thresh beta_prev) < tol then beta_thresh
      else optimize (iter + 1) beta_thresh
  in
  
  let beta_lasso = optimize 0 !beta in
  sub x_j (matmul x_minus_j beta_lasso)

let debiased_projection x theta theta_star config =
  let p = size x 1 in
  let theta_debiased = zeros [p] in
  
  for j = 0 to p-1 do
    let r_j = compute_residuals x j config.lambda_x in
    let x_j = select x 1 j in
    
    (* Debiasing correction term *)
    let correction = div 
      (dot r_j (sub (matmul x theta) (matmul x theta_star)))
      (dot r_j x_j) in
      
    (* Update debiased estimate *)
    let theta_j_db = add (select theta_star 0 j) correction in
    tensor_set theta_debiased [j] theta_j_db
  done;
  
  theta_debiased

let debiased_projection_distributed data theta theta_star config =
  let p = data.p in
  let theta_debiased = zeros [p] in
  
  for j = 0 to p-1 do
    let r_j = compute_residuals x j config.lambda_x in
    let x_j = select x 1 j in
    
    let correction = div 
      (dot r_j (sub (matmul data.xtx theta) (matmul data.xtx theta_star)))
      (dot r_j x_j) in
      
    let theta_j_db = add (select theta_star 0 j) correction in
    tensor_set theta_debiased [j] theta_j_db
  done;
  
  theta_debiased