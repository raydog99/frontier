open Torch

module Lasso = struct
  let soft_threshold x lambda =
    sign x * max (abs x - lambda) (zeros_like x)
    
  let coordinate_descent ~x ~y ~lambda ~max_iter =
    let n, p = (size x 0), (size x 1) in
    let beta = zeros [p] in
    let x_squared = sum (x * x) ~dim:[0] in
    
    for _ = 1 to max_iter do
      for j = 0 to p - 1 do
        let xj = select x ~dim:1 ~index:j in
        let r = y - (mm x beta) in
        let beta_j = beta.(.j) in
        r += xj * beta_j;
        
        let coordinate = dot xj r in
        let update = soft_threshold coordinate lambda in
        beta.(.j) <- update / x_squared.(.j)
      done
    done;
    beta
end

module RidgeRegression = struct
  let condition_number m =
    let open Tensor in
    let svd = svd m in
    let s = third_of_triple svd in
    let max_s = max s in
    let min_s = min s in
    float_value max_s /. float_value min_s
    
  let solve_system a b =
    let open Tensor in
    try
      Some (gesv ~a ~b |> snd)
    with _ -> None
    
  let estimate ~x ~y ~lambda =
    let open Tensor in
    let xt = transpose x ~dim0:0 ~dim1:1 in
    let n, p = size x 0, size x 1 in
    
    (* Add regularization *)
    let gram = matmul xt x + (eye p * lambda) in
    let rhs = matmul xt y in
    
    (* Check condition number *)
    let cond = condition_number gram in
    if cond > 1e10 then begin
      (* Use SVD for numerical stability *)
      let u, s, v = svd gram in
      let s_inv = div_scalar (ones_like s) (s + lambda) in
      let vt = transpose v ~dim0:0 ~dim1:1 in
      let ut = transpose u ~dim0:0 ~dim1:1 in
      matmul (matmul (matmul v (diag s_inv)) ut) rhs
    end else
      (* Use direct solve if well-conditioned *)
      match solve_system gram rhs with
      | Some solution -> solution
      | None ->
          (* Fallback to gradient descent *)
          let beta = zeros [p] in
          let lr = 0.01 in
          let max_iter = 1000 in
          let tol = 1e-6 in
          
          let rec iterate iter prev_loss =
            if iter >= max_iter then beta
            else begin
              let pred = matmul x beta in
              let loss = mean (pow (sub y pred) (scalar 2.)) in
              let diff = abs_float (float_value (sub loss prev_loss)) in
              
              if diff < tol then beta
              else begin
                let grad = matmul xt (sub pred y) in
                beta -= grad * lr;
                iterate (iter + 1) loss
              end
            end
          in
          iterate 0 (zeros [1])

  let cross_validate ~x ~y ~lambda_grid ~k_folds =
    let open Tensor in
    let n = size x 0 in
    let fold_size = n / k_folds in
    
    let errors = List.map (fun lambda ->
      let fold_errors = List.init k_folds (fun fold ->
        (* Create train/validation split *)
        let val_start = fold * fold_size in
        let val_end = min (val_start + fold_size) n in
        
        let x_val = narrow x ~dim:0 ~start:val_start ~length:(val_end - val_start) in
        let y_val = narrow y ~dim:0 ~start:val_start ~length:(val_end - val_start) in
        
        let x_train = cat [
          narrow x ~dim:0 ~start:0 ~length:val_start;
          narrow x ~dim:0 ~start:val_end ~length:(n - val_end)
        ] ~dim:0 in
        let y_train = cat [
          narrow y ~dim:0 ~start:0 ~length:val_start;
          narrow y ~dim:0 ~start:val_end ~length:(n - val_end)
        ] ~dim:0 in
        
        (* Train and evaluate *)
        let beta = estimate ~x:x_train ~y:y_train ~lambda in
        let preds = matmul x_val beta in
        mean (pow (sub y_val preds) (scalar 2.))
      ) in
      (lambda, Stats.mean (stack fold_errors))
    ) lambda_grid in
    
    (* Return lambda with minimum error *)
    List.fold_left (fun (best_lambda, min_error) (lambda, error) ->
      if float_value error < float_value min_error then
        (lambda, error)
      else
        (best_lambda, min_error)
    ) (List.hd lambda_grid, tensor 1e10) errors
    |> fst
end