open Torch
open Type

let sparse_projection x theta lambda =
  let n = float_of_int (size x 0) in
  let p = size x 1 in
  
  (* Initialize optimization *)
  let u = zeros [p] in
  let max_iter = 1000 in
  let tol = 1e-6 in
  
  (* Proximal gradient descent *)
  let rec optimize iter u_prev =
    if iter >= max_iter then u_prev
    else
      (* Gradient step *)
      let x_theta = matmul x theta in
      let x_u = matmul x u_prev in
      let grad = scalar_mul (2. /. n) (matmul (transpose2D x ~dim0:0 ~dim1:1) (sub x_u x_theta)) in
      let u_next = sub u_prev (scalar_mul 0.01 grad) in
      
      (* Soft thresholding *)
      let u_thresh = map (fun x -> 
        let abs_x = abs_float x in
        if abs_x <= lambda then 0.
        else copysign (abs_x -. lambda) x
      ) u_next in
      
      (* Check convergence *)
      let diff = norm (sub u_thresh u_prev) in
      if diff < tol then u_thresh
      else optimize (iter + 1) u_thresh
  in
  optimize 0 u

let sparse_projection_distributed data theta lambda =
  let n = float_of_int data.n in
  let p = data.p in
  
  let u = zeros [p] in
  let max_iter = 1000 in
  let tol = 1e-6 in
  
  let rec optimize iter u_prev =
    if iter >= max_iter then u_prev
    else
      (* Gradient using sufficient statistics *)
      let grad = scalar_mul (2. /. n) (sub 
        (matmul data.xtx u_prev)
        (matmul data.xtx theta)) in
      let u_next = sub u_prev (scalar_mul 0.01 grad) in
      
      let u_thresh = map (fun x -> 
        let abs_x = abs_float x in
        if abs_x <= lambda then 0.
        else copysign (abs_x -. lambda) x
      ) u_next in
      
      let diff = norm (sub u_thresh u_prev) in
      if diff < tol then u_thresh
      else optimize (iter + 1) u_thresh
  in
  optimize 0 u