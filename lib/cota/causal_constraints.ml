open Torch
open Types

let truncated_factorization scm intervention samples =
  (* Get parents for intervened variables *)
  let parents = List.flatten (Array.to_list (Array.map (fun var ->
    List.filter_map (fun (p, c) ->
      if c = var then Some p else None
    ) scm.graph
  ) intervention.variables)) in
  
  (* Compute conditional probabilities *)
  let cond_probs = List.map (fun parent ->
    let parent_samples = Tensor.slice samples [parent] in
    let child_idx = List.find_map (fun (p, c) ->
      if p = parent then Some c else None
    ) scm.graph |> Option.get in
    let child_samples = Tensor.slice samples [child_idx] in
    
    (* Compute P(child|parent) *)
    let joint = Tensor.stack [parent_samples; child_samples] ~dim:1 in
    let unique_parents = Tensor.unique parent_samples in
    
    List.map (fun p ->
      let mask = Tensor.(eq parent_samples p) in
      let filtered = Tensor.masked_select child_samples mask in
      let counts = Tensor.bincount filtered in
      Tensor.(div counts (sum counts |> add (float 1e-10)))
    ) (Tensor.to_float1_exn unique_parents)
  ) parents in
  
  (* Multiply conditional probabilities *)
  List.fold_left (fun acc probs ->
    List.fold_left Tensor.mul acc probs
  ) (Tensor.ones [1]) cond_probs

let do_calculus_distance plan1 plan2 intervention1 intervention2 =
  (* Get marginals *)
  let m1 = Tensor.(sum plan1.plan ~dim:[1]) in
  let m2 = Tensor.(sum plan2.plan ~dim:[1]) in
  
  (* Get normalizing terms *)
  let z1 = truncated_factorization plan1.scm intervention1 m1 in
  let z2 = truncated_factorization plan2.scm intervention2 m2 in
  
  (* Compute normalized distributions *)
  let p1 = Tensor.(div m1 z1) in
  let p2 = Tensor.(div m2 z2) in
  
  (* Compute KL divergence *)
  let eps = Tensor.full [1] 1e-10 in
  Tensor.(sum (mul p1 (log (div (add p1 eps) (add p2 eps)))))