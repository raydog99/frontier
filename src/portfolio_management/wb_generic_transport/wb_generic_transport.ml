open Torch

type space = {
  dim: int;
  metric: Tensor.t -> Tensor.t -> float;
  is_compact: bool;
}

type measure = {
  weights: Tensor.t;
  points: Tensor.t;
  space: space;
}

type cost = {
  source_space: space;
  target_space: space;
  fn: Tensor.t -> Tensor.t -> Tensor.t;
  is_continuous: bool;
}

type transport_plan = {
  plan: Tensor.t;
  source: Tensor.t;
  target: Tensor.t;
}

type coupling = {
  measure: Tensor.t;
  marginals: Tensor.t list;
  support: Tensor.t list;
}

let logsumexp x ~dim =
  let max_x = Tensor.max x ~dim ~keepdim:true |> fst in
  let shifted = Tensor.sub x max_x in
  let sum_exp = Tensor.sum (Tensor.exp shifted) ~dim ~keepdim:true in
  Tensor.(log sum_exp + max_x)

let matrix_sqrt mat =
  let e, v = Tensor.symeig mat ~eigenvectors:true in
  let sqrt_e = Tensor.sqrt e in
  Tensor.(mm (mm v (Tensor.diag sqrt_e)) (transpose v ~dim0:0 ~dim1:1))

let is_positive_definite mat =
  try
    let _ = Tensor.cholesky mat in true
  with _ -> false

let nearest_pd mat =
  let symm = Tensor.((mat + transpose mat ~dim0:0 ~dim1:1) * scalar 0.5) in
  let eigvals, eigvecs = Tensor.symeig symm ~eigenvectors:true in
  let min_eig = Tensor.min eigvals |> fst |> Tensor.to_float0_exn in
  if min_eig > 0. then symm
  else
    let shifted_eigvals = Tensor.(max eigvals (ones_like eigvals * scalar 1e-6)) in
    let result = Tensor.(mm (mm eigvecs (diag shifted_eigvals)) 
                           (transpose eigvecs ~dim0:0 ~dim1:1)) in
    Tensor.((result + transpose result ~dim0:0 ~dim1:1) * scalar 0.5)

let compute_cost_matrix ~source ~target ~cost =
  let m, n = 
    let dims_s = Tensor.size source in
    let dims_t = Tensor.size target in
    List.nth dims_s 0, List.nth dims_t 0 in
  let matrix = Tensor.zeros [m; n] in
  
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let c = cost (Tensor.select source 0 i) (Tensor.select target 0 j) in
      Tensor.set matrix [|i; j|] (Tensor.to_float0_exn c)
    done
  done;
  matrix

let verify_optimality plan cost_matrix source_weights target_weights =
  let marginal_x = Tensor.sum plan ~dim:[1] in
  let marginal_y = Tensor.sum plan ~dim:[0] in
  Tensor.allclose marginal_x source_weights ~rtol:1e-5 &&
  Tensor.allclose marginal_y target_weights ~rtol:1e-5

let compute_b ~costs ~points ~weights =
  let dim = List.hd points |> Tensor.size |> List.tl |> List.hd in
  let init = Tensor.mean (Tensor.stack points 0) ~dim:[0] in
  
  let rec optimize x iter =
    if iter > 100 then x
    else
      let total_cost = List.fold_left3 (fun acc cost point weight ->
        let c = cost.fn x point in
        Tensor.(acc + c * weight)
      ) (Tensor.zeros []) costs points weights in
      
      let grad = Tensor.grad total_cost in
      let x' = Tensor.(x - grad * scalar 0.01) in
      
      if Tensor.norm (Tensor.sub x' x) < 1e-6 then x'
      else optimize x' (iter + 1)
  in
  optimize init 0

let solve_sinkhorn ~cost_matrix ~source_weights ~target_weights ~epsilon =
  let kernel = Tensor.(exp (neg cost_matrix / scalar epsilon)) in
  let m, n = List.nth (Tensor.size cost_matrix) 0, List.nth (Tensor.size cost_matrix) 1 in
  
  let rec iterate u v iter =
    if iter > 1000 then (u, v)
    else
      let u' = Tensor.(source_weights / (kernel * v)) in
      let v' = Tensor.(target_weights / (transpose kernel ~dim0:0 ~dim1:1 * u)) in
      
      if iter mod 10 = 0 && 
         Tensor.allclose u u' ~rtol:1e-6 && 
         Tensor.allclose v v' ~rtol:1e-6
      then (u', v')
      else iterate u' v' (iter + 1)
  in
  
  let u0 = Tensor.ones [m] in
  let v0 = Tensor.ones [n] in
  let u, v = iterate u0 v0 0 in
  Tensor.(kernel * (u @@ v))

let solve_exact ~cost_matrix ~source_weights ~target_weights =
  let n = Tensor.size cost_matrix |> List.hd in
  let m = Tensor.size cost_matrix |> List.tl |> List.hd in
  
  (* Initialize with product measure *)
  let init_plan = Tensor.(outer source_weights target_weights) in
  
  let rec iterate plan iter =
    if iter > 100 then plan
    else
      (* Compute gradient *)
      let grad = Tensor.(cost_matrix + log plan) in
      
      (* Project onto transport polytope *)
      let step = Tensor.(plan - grad * scalar 0.1) in
      let new_plan = Tensor.relu step in
      
      (* Normalize to satisfy marginal constraints *)
      let scale_x = Tensor.(div source_weights (sum new_plan ~dim:[1])) in
      let scaled_x = Tensor.(mul new_plan (unsqueeze scale_x 1)) in
      let scale_y = Tensor.(div target_weights (sum scaled_x ~dim:[0])) in
      let scaled = Tensor.(mul scaled_x (unsqueeze scale_y 0)) in
      
      if Tensor.allclose plan scaled ~rtol:1e-6 then scaled
      else iterate scaled (iter + 1)
  in
  iterate init_plan 0

module Fixed_point = struct
  let compute_next measure targets costs =
    (* Optimal transport plans *)
    let plans = List.map2 (fun target cost ->
      let cost_matrix = compute_cost_matrix 
        ~source:measure.points 
        ~target:target.points 
        ~cost:cost.fn in
      solve_sinkhorn 
        ~cost_matrix 
        ~source_weights:measure.weights
        ~target_weights:target.weights
        ~epsilon:1e-3
    ) targets costs in
    
    (* Ground barycentre *)
    let new_points = compute_b
      ~costs
      ~points:(List.map (fun t -> t.points) targets)
      ~weights:(List.map (fun p -> Tensor.sum p ~dim:[1]) plans) in
    
    { measure with points = new_points }

  let iterate ~init ~targets ~costs ~max_iter =
    let rec loop measure iter =
      if iter >= max_iter then measure
      else
        let next = compute_next measure targets costs in
        if Tensor.allclose measure.points next.points ~rtol:1e-6
        then next
        else loop next (iter + 1)
    in
    loop init 0

  let check_convergence sequence =
    match sequence with
    | [] | [_] -> false
    | m1 :: rest ->
        List.for_all (fun m2 ->
          Tensor.allclose m1.points m2.points ~rtol:1e-6
        ) rest
end

(* Discrete iteration of G *)
let iterate_g ~init ~targets ~costs ~max_iter =
  let rec iterate measure t =
    if t >= max_iter then measure
    else
      (* Optimal plans *)
      let plans = List.map2 (fun target cost ->
        let cost_matrix = compute_cost_matrix 
          ~source:measure.points 
          ~target:target.points 
          ~cost:cost.fn in
        solve_sinkhorn 
          ~cost_matrix 
          ~source_weights:measure.weights
          ~target_weights:target.weights
          ~epsilon:1e-3
      ) targets costs in
      
      (* Update support points *)
      let new_points = compute_b
        ~costs
        ~points:(List.map (fun t -> t.points) targets)
        ~weights:(List.map (fun p -> Tensor.sum p ~dim:[1]) plans) in
      
      let next = { measure with points = new_points } in
      
      if Fixed_point.check_convergence [measure; next] then next
      else iterate next (t + 1)
  in
  iterate init 0

(* Discrete iteration of H *)
let iterate_h ~init ~targets ~costs ~max_iter =
  let rec iterate measure t =
    if t >= max_iter then measure
    else
      (* Optimal plans *)
      let plans = List.map2 (fun target cost ->
        let cost_matrix = compute_cost_matrix 
          ~source:measure.points 
          ~target:target.points 
          ~cost:cost.fn in
        solve_sinkhorn 
          ~cost_matrix 
          ~source_weights:measure.weights
          ~target_weights:target.weights
          ~epsilon:1e-3
      ) targets costs in
      
      (* Barycentric projection *)
      let new_points = List.fold_left2 (fun acc plan target ->
        let weights = Tensor.sum plan ~dim:[1] in
        let proj = Tensor.(mm plan target.points / unsqueeze weights 1) in
        Tensor.(acc + proj)
      ) (Tensor.zeros_like init.points) plans targets in
      
      let next = { measure with points = new_points } in
      
      if Fixed_point.check_convergence [measure; next] then next
      else iterate next (t + 1)
  in
  iterate init 0

module GMM = struct
  type gaussian = {
    mean: Tensor.t;
    covariance: Tensor.t;
  }

  type gmm = {
    components: gaussian list;
    weights: Tensor.t;
  }

  (* Bures-Wasserstein distance *)
  let bures_wasserstein g1 g2 =
    let mean_term = Tensor.norm2 (Tensor.sub g1.mean g2.mean) in
    let s1_sqrt = matrix_sqrt g1.covariance in
    let product = Tensor.(mm (mm s1_sqrt g2.covariance) s1_sqrt) in
    let sqrt_prod = matrix_sqrt product in
    
    let cov_term = Tensor.(
      trace g1.covariance + 
      trace g2.covariance - 
      (scalar 2.0 * trace sqrt_prod)
    ) in
    
    Tensor.(mean_term + cov_term)

  let fixed_point_iteration s components weights =
    let weighted_sum = List.fold_left2 (fun acc comp w ->
      let s_sqrt = matrix_sqrt s in
      let term = Tensor.(mm (mm s_sqrt comp.covariance) s_sqrt) in
      let sqrt_term = matrix_sqrt term in
      Tensor.(acc + sqrt_term * scalar w)
    ) (Tensor.zeros_like s) components weights in
    
    let s_inv_sqrt = Tensor.inverse (matrix_sqrt s) in
    Tensor.(mm (mm s_inv_sqrt (pow weighted_sum (scalar 2.0))) s_inv_sqrt)

  let compute_barycentre gmms weights max_iter =
    (* Initialize with weighted average *)
    let init_mean = List.fold_left2 (fun acc gmm w ->
      let mean = List.fold_left (fun m g -> Tensor.(m + g.mean)) 
        (Tensor.zeros_like (List.hd (List.hd gmms).components).mean)
        gmm.components in
      Tensor.(acc + mean * scalar w)
    ) (Tensor.zeros_like (List.hd (List.hd gmms).components).mean)
      gmms weights in
    
    let init_cov = List.fold_left2 (fun acc gmm w ->
      let cov = List.fold_left (fun c g -> Tensor.(c + g.covariance))
        (Tensor.zeros_like (List.hd (List.hd gmms).components).covariance)
        gmm.components in
      Tensor.(acc + cov * scalar w)
    ) (Tensor.zeros_like (List.hd (List.hd gmms).components).covariance)
      gmms weights in
    
    (* Fixed point iteration *)
    let rec iterate s iter =
      if iter >= max_iter then s
      else
        let s_next = fixed_point_iteration s 
          (List.flatten (List.map (fun g -> g.components) gmms))
          weights in
        
        if Tensor.allclose s s_next ~rtol:1e-5 then s_next
        else iterate s_next (iter + 1)
    in
    
    let final_cov = iterate init_cov 0 in
    { mean = init_mean; covariance = final_cov }
end