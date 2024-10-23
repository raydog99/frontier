open Torch
open Types
open Matrix_ops

let compute_optimality_gradient ~omega ~sigma ~gamma ~sigma_sq =
  let dim = Tensor.size omega |> List.hd in
  let inv_omega = Tensor.inverse omega in
  let scaled_sigma = Tensor.mul_scalar sigma (gamma *. sigma_sq /. float_of_int dim) in
  let term1 = Tensor.mm scaled_sigma inv_omega in
  let term2 = Tensor.mm term1 scaled_sigma in
  Tensor.sub term2 (Tensor.eye dim)

let verify_pd_hessian ~point ~gamma ~sigma_sq =
  let dim = Tensor.size point |> List.hd in
  let epsilon = 1e-6 in
  
  (* Generate random directions *)
  let directions = List.init 10 (fun _ -> 
    let d = Tensor.randn [dim; dim] in
    let dt = Tensor.transpose d ~dim0:0 ~dim1:1 in
    Tensor.add d dt |> fun x -> 
    Tensor.div_scalar x (2. *. (Tensor.norm x |> Tensor.to_float0_exn))
  ) in
  
  (* Check positive definiteness along directions *)
  List.for_all (fun direction ->
    let next_point = Tensor.add point (Tensor.mul_scalar direction epsilon) in
    let prev_point = Tensor.add point (Tensor.mul_scalar direction (-.epsilon)) in
    let grad_diff = Tensor.sub 
      (compute_optimality_gradient ~omega:next_point ~sigma:point ~gamma ~sigma_sq)
      (compute_optimality_gradient ~omega:prev_point ~sigma:point ~gamma ~sigma_sq) in
    let hess_dir = Tensor.div_scalar grad_diff (2. *. epsilon) in
    Tensor.dot direction hess_dir |> Tensor.to_float0_exn > 0.
  ) directions

let compute_optimal_risk ~omega ~sigma ~gamma ~sigma_sq =
  let dim = Tensor.size omega |> List.hd in
  let lambda = gamma *. sigma_sq in
  
  let term1 = Tensor.trace sigma |> Tensor.to_float0_exn in
  let term2 = lambda *. float_of_int dim in
  let term3 = gamma *. sigma_sq in
  
  term1 +. term2 +. term3