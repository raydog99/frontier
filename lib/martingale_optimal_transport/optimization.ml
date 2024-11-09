open Torch

(* Interior point method parameters *)
type barrier_params = {
  mu: float;
  beta: float;
  max_inner_iter: int;
  tolerance: float;
}

(* Constraint types *)
type constraint_type =
  | Martingale
  | Marginal
  | Support
  | Moment
  | CustomLinear of (Tensor.t -> float)

(* Interior point solver *)
let solve_interior_point problem params =
  let n = MOT.get_dimension problem in
  
  (* Initialize primal and dual variables *)
  let x = Tensor.ones [|n; n|] |>
    Tensor.div_scalar (float_of_int (n * n)) in
  let y = Tensor.zeros [|n|] in
  let z = Tensor.ones [|n * n|] in
  
  let rec newton_iterate x y z iter =
    if iter >= params.max_inner_iter then x
    else
      (* Form KKT system *)
      let barrier = Tensor.sum 
        (Tensor.neg (Tensor.log x)) |>
        Tensor.float_value in
      
      (* Compute Newton direction *)
      let dx, dy = Optimization.InteriorPoint.compute_newton_direction
        problem x y z params.mu in
      
      (* Line search *)
      let alpha = Optimization.InteriorPoint.line_search
        problem x y z dx dy in
      
      (* Update variables *)
      let new_x = Tensor.add x 
        (Tensor.mul_scalar dx alpha) in
      let new_y = Tensor.add y 
        (Tensor.mul_scalar dy alpha) in
      let new_z = z in  (* Dual update for inequality constraints *)
      
      (* Check convergence *)
      let gap = Tensor.dot x z |> Tensor.float_value in
      if gap < params.tolerance then new_x
      else newton_iterate new_x new_y new_z (iter + 1)
  in
  
  newton_iterate x y z 0

(* Entropic regularization solver *)
let solve_entropic problem gamma max_iter =
  let marginals = MOT.get_marginals problem in
  let n = Array.length marginals in
  
  (* Initialize transport plan *)
  let plan = Tensor.ones [|n; n|] |>
    Tensor.div_scalar (float_of_int (n * n)) in
  
  let rec sinkhorn_iterate plan iter =
    if iter >= max_iter then plan
    else
      (* Project onto marginal constraints *)
      let proj_margins = Array.fold_left2
        (fun p mu i ->
          let density = DiscreteMeasure.density mu |> 
            Option.get in
          let scale = Tensor.div density
            (Tensor.sum p ~dim:[|i|]) in
          Tensor.mul p 
            (Tensor.unsqueeze scale ~dim:(1-i)))
        plan marginals [|0; 1|] in
      
      (* Add entropic regularization *)
      let reg_plan = Tensor.mul
        proj_margins
        (Tensor.exp (Tensor.div_scalar
           (MOT.evaluate problem proj_margins)
           (-.gamma))) in
      
      (* Check convergence *)
      let diff = Tensor.norm
        (Tensor.sub reg_plan plan)
        ~p:2 ~dim:[|0; 1|] |>
        Tensor.float_value in
      
      if diff < 1e-6 then reg_plan
      else sinkhorn_iterate reg_plan (iter + 1)
  in
  
  sinkhorn_iterate plan 0

(* Project onto constraints *)
let project_constraints plan constraints tolerance =
  Array.fold_left
    (fun p c -> match c with
      | Martingale ->
          (* Project onto martingale constraint *)
          let curr_vals = Tensor.narrow p ~dim:0 
            ~start:0 ~length:1 in
          let next_vals = Tensor.narrow p ~dim:0 
            ~start:1 ~length:1 in
          let diff = Tensor.sub next_vals curr_vals in
          
          let proj = Tensor.where
            (Tensor.le (Tensor.abs diff) 
               (Tensor.scalar_tensor tolerance))
            p (Tensor.zeros_like p) in
          proj
      | Marginal ->
          (* Project onto marginal constraints *)
          let row_sums = Tensor.sum p ~dim:[|1|] in
          let col_sums = Tensor.sum p ~dim:[|0|] in
          let row_scale = Tensor.div 
            (Tensor.ones_like row_sums)
            row_sums in
          let col_scale = Tensor.div
            (Tensor.ones_like col_sums)
            col_sums in
          
          Tensor.mul
            (Tensor.mul p 
               (Tensor.unsqueeze row_scale ~dim:1))
            (Tensor.unsqueeze col_scale ~dim:0)
      | Support ->
          (* Project onto support constraint *)
          Tensor.maximum p (Tensor.zeros_like p)
      | Moment ->
          (* Project onto moment constraints *)
          let mean = Tensor.mean p in
          let std = Tensor.std p ~unbiased:true in
          Tensor.div (Tensor.sub p mean) std
      | CustomLinear f ->
          (* Project onto custom linear constraint *)
          let val_f = f p in
          if abs_float val_f > tolerance then
            let grad = Tensor.grad_of_fn f p in
            let step = val_f /. 
              (Tensor.norm grad ~p:2 ~dim:[|0|] |>
               Tensor.float_value) in
            Tensor.sub p (Tensor.mul_scalar grad step)
          else p)
    plan constraints