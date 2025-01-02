open Torch
open Types

let entropy plan =
  let eps = Tensor.full [1] 1e-10 in
  let safe_plan = Tensor.(plan + eps) in
  Tensor.(neg (sum (mul safe_plan (log safe_plan))))

let cost_matrix source_samples target_samples interventions omega_map =
  let n_source = (Tensor.shape source_samples |> List.hd) in
  let n_target = (Tensor.shape target_samples |> List.hd) in
  let n_interventions = List.length interventions in
  
  let cost = Tensor.full [n_source; n_target] (float_of_int n_interventions) in
  
  List.fold_left (fun cost_acc i ->
    let i' = omega_map i in
    let comp_matrix = Tensor.init [n_source; n_target] (fun idx ->
      let s = idx / n_target in
      let t = idx mod n_target in
      let source_comp = is_compatible (Tensor.slice1 source_samples [s]) i in
      let target_comp = is_compatible (Tensor.slice1 target_samples [t]) i' in
      if source_comp && target_comp then -1.0 else 0.0
    ) in
    Tensor.(cost_acc + comp_matrix)
  ) cost interventions

let sinkhorn cost mu nu epsilon =
  let max_iter = 100 in
  let threshold = 1e-6 in
  let n_source, n_target = match Tensor.shape cost with
    | [a; b] -> a, b
    | _ -> failwith "Invalid cost matrix shape" in
  
  (* Initialize Gibbs kernel *)
  let k = Tensor.(exp (neg (div cost (float epsilon)))) in
  
  (* Initialize scaling vectors *)
  let u = Tensor.ones [n_source] in
  let v = Tensor.ones [n_target] in
  
  let rec iterate u v iter =
    if iter >= max_iter then 
      Tensor.(mul (mul (reshape u ~shape:[-1; 1]) k) (reshape v ~shape:[1; -1]))
    else
      (* Update u *)
      let u_new = Tensor.(div mu (matmul k v)) in
      (* Update v *)
      let v_new = Tensor.(div nu (matmul (transpose k ~dim0:0 ~dim1:1) u_new)) in
      (* Check convergence *)
      let err = Tensor.(abs (sum (sub u_new u)) |> item) in
      if err < threshold then 
        Tensor.(mul (mul (reshape u_new ~shape:[-1; 1]) k) (reshape v_new ~shape:[1; -1]))
      else iterate u_new v_new (iter + 1)
  in
  
  iterate u v 0

let gradient_step plan grad lr =
  let updated = Tensor.(sub plan (mul_scalar grad lr)) in
  (* Project back to transport polytope *)
  let row_sums = Tensor.(sum updated ~dim:[1] |> reshape ~shape:[-1; 1]) in
  let col_sums = Tensor.(sum updated ~dim:[0] |> reshape ~shape:[1; -1]) in
  Tensor.(div updated (add (add row_sums col_sums) (full [1] 1e-10)))