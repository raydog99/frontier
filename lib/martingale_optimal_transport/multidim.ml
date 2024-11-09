open Torch

type dimension_spec = {
  spatial_dims: int array;
  time_dims: int;
  batch_size: int option;
}

type solver_config = {
  dimension_spec: dimension_spec;
  regularization: float;
  max_iter: int;
  tolerance: float;
  batch_size: int option;
}

(* Grid creation for multi-dimensional problems *)
let create_grid spec bounds n_points =
  let create_dim_grid (lower, upper) n =
    Tensor.linspace lower upper n in
  
  let spatial_grids = Array.map2
    (fun (l, u) dim -> create_dim_grid (l, u) n_points)
    bounds spec.spatial_dims in
  
  let meshgrid = Tensor.meshgrid (Array.to_list spatial_grids) in
  Tensor.stack (Array.of_list meshgrid) ~dim:0

(* Multi-dimensional marginal computation *)
let compute_marginals tensor dims =
  let all_dims = Array.init (Array.length (Tensor.shape tensor)) 
    (fun i -> i) in
  let keep_dims = Array.filter
    (fun d -> not (Array.mem d dims))
    all_dims in
  
  Tensor.mean tensor ~dim:(Array.to_list dims)

(* Tucker decomposition for tensor transport plan *)
let tucker_decompose plan rank =
  let dims = Tensor.shape plan in
  let n_dims = Array.length dims in
  
  (* Initialize factors *)
  let factors = Array.init n_dims (fun i ->
    Tensor.randn [|dims.(i); rank|]) in
  
  (* ALS iterations *)
  let rec als_iterate factors iter =
    if iter >= 100 then factors
    else
      let new_factors = Array.mapi (fun i factor ->
        (* Matricize tensor *)
        let matrix = Tensor.reshape plan [|dims.(i); -1|] in
        
        (* Compute Kronecker product of other factors *)
        let other_factors = Array.concat [
          Array.sub factors 0 i;
          Array.sub factors (i+1) (n_dims - i - 1)
        ] in
        let kron = Array.fold_left
          (fun acc f -> Tensor.kron acc f)
          (Tensor.eye rank)
          other_factors in
        
        (* Update factor *)
        let update = Tensor.matmul matrix kron in
        let u, s, v = Tensor.svd update in
        Tensor.narrow u ~dim:1 ~start:0 ~length:rank
      ) factors in
      
      let change = Array.map2
        (fun old_f new_f -> 
          Tensor.norm (Tensor.sub old_f new_f) 
            ~p:2 ~dim:[|0; 1|] |>
          Tensor.float_value)
        factors new_factors |>
        Array.fold_left max 0.0 in
      
      if change < 1e-6 then new_factors
      else als_iterate new_factors (iter + 1)
  in
  
  als_iterate factors 0

(* Multi-dimensional Sinkhorn algorithm *)
let multidim_sinkhorn problem config =
  let marginals = MOT.get_marginals problem in
  let n_marginals = Array.length marginals in
  
  (* Initialize dual variables *)
  let dual_vars = Array.map (fun m ->
    Tensor.zeros [|Tensor.shape 
      (DiscreteMeasure.support m).(0)|]
  ) marginals in
  
  let rec iterate dual_vars iter =
    if iter >= config.max_iter then dual_vars
    else
      (* Update each dual variable *)
      let new_vars = Array.mapi (fun i var ->
        let kernel = Tensor.exp (Tensor.neg (
          MOT.evaluate problem (
            let next_idx = (i+1) mod n_marginals in
            Tensor.cat [
              DiscreteMeasure.support marginals.(i);
              DiscreteMeasure.support marginals.(next_idx)
            ] ~dim:1))) in
        
        let scaled_kernel = Tensor.mul kernel 
          (Tensor.exp var) in
        
        Tensor.log (Tensor.div
          (DiscreteMeasure.density marginals.(i) |> Option.get)
          (Tensor.sum scaled_kernel ~dim:1))
      ) dual_vars in
      
      let max_change = Array.map2
        (fun old_v new_v ->
          Tensor.max (Tensor.abs (Tensor.sub old_v new_v)) |>
          Tensor.float_value)
        dual_vars new_vars |>
        Array.fold_left max 0.0 in
      
      if max_change < config.tolerance then new_vars
      else iterate new_vars (iter + 1)
  in
  
  iterate dual_vars 0

(* Main solver *)
let solve problem config =
  let rank = Array.fold_left min max_int 
    (Array.map (fun m ->
      (Tensor.shape (DiscreteMeasure.support m)).(0))
      (MOT.get_marginals problem)) in
  
  (* Get initial solution using tensor decomposition *)
  let initial_factors = tucker_decompose 
    (Tensor.randn (Array.map (fun m ->
      (Tensor.shape (DiscreteMeasure.support m)).(0))
      (MOT.get_marginals problem)))
    rank in
  
  (* Refine using Sinkhorn *)
  let dual_vars = multidim_sinkhorn problem 
    {config with max_iter = 1000} in
  
  (* Combine solutions *)
  let tensor_plan = Array.fold_left2
    (fun acc factor dual ->
      let scaled_factor = Tensor.mul factor 
        (Tensor.exp dual) in
      Tensor.matmul acc (Tensor.transpose scaled_factor 
        ~dim0:0 ~dim1:1))
    (Array.get initial_factors 0)
    (Array.sub initial_factors 1 
      (Array.length initial_factors - 1))
    dual_vars in
  
  (* Project onto constraints *)
  Solver.project_constraints problem tensor_plan