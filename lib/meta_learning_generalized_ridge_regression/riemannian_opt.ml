open Torch
open Types
open Matrix_ops

(* Retraction map for positive definite matrices *)
let retract ~point ~direction =
  let inv_point = Tensor.inverse point in
  let quad_term = Tensor.mm (Tensor.mm direction inv_point) direction in
  let quad_term = Tensor.mul_scalar quad_term 0.5 in
  let result = Tensor.add (Tensor.add point direction) quad_term in
  nearest_positive_definite result

(* Riemannian metric for positive definite matrices *)
let metric ~point ~tangent1 ~tangent2 =
  let inv_point = Tensor.inverse point in
  let term1 = Tensor.mm tangent1 inv_point in
  let term2 = Tensor.mm tangent2 inv_point in
  Tensor.mm term1 term2 |> Tensor.trace |> Tensor.to_float0_exn

(* Vector transport along retraction *)
let vector_transport ~point ~direction ~target =
  let point_sqrt = matrix_power point 0.5 in
  let point_inv_sqrt = matrix_power point (-0.5) in
  let term = Tensor.mm (Tensor.mm point_sqrt direction) point_inv_sqrt in
  Tensor.mm (Tensor.mm target term) (Tensor.inverse target)

(* Line search with Armijo condition *)
let rec line_search ~objective ~point ~direction ~step ~beta ~sigma =
  if step < 1e-10 then step
  else
    let new_point = retract ~point ~direction:(Tensor.mul_scalar direction step) in
    let decrease = sigma *. step *. 
                  (Tensor.dot direction (Tensor.neg direction) |> Tensor.to_float0_exn) in
    if objective new_point <= objective point +. decrease 
    then step
    else line_search ~objective ~point ~direction ~step:(step *. beta) ~beta ~sigma

(* Riemannian optimization *)
let optimize ~objective ~gradient ~init ~params =
  let dim = Tensor.size init |> List.hd in
  
  let rec optimize_iter point velocity step iter =
    if iter >= params.max_iter then point
    else
      let grad = gradient point in
      let grad_norm = frobenius_norm grad in
      
      if grad_norm < params.tol then point
      else
        (* Compute search direction with momentum *)
        let direction = 
          if iter = 0 then Tensor.neg grad
          else
            let momentum_term = vector_transport ~point:velocity ~direction:grad ~target:point in
            Tensor.add (Tensor.neg grad) 
              (Tensor.mul_scalar momentum_term params.beta)
        in
        
        (* Line search *)
        let step_size = line_search 
          ~objective 
          ~point 
          ~direction 
          ~step:params.step_size 
          ~beta:0.5 
          ~sigma:1e-4 in
        
        (* Update *)
        let new_point = retract ~point ~direction:(Tensor.mul_scalar direction step_size) in
        optimize_iter new_point direction step_size (iter + 1)
  in
  
  optimize_iter init (Tensor.zeros [dim; dim]) params.step_size 0