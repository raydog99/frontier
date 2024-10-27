type nuisance_functions = {
  m_hat: float array;
  f_hat: float array;
  h_hat: float array;
  v_hat: float array array;
}

module RandomForest = struct
  type regression_tree = {
    root: Types.node;
    responses: float array;
  }

  type forest = {
    trees: regression_tree array;
    n_trees: int;
  }

  let predict_tree tree x z =
    let rec traverse node =
      match (node.split_var, node.split_point) with
      | (None, None) -> 
          let sum = ref 0.0 in
          let count = ref 0 in
          Array.iter (fun idx ->
            sum := !sum +. tree.responses.(idx);
            count := !count + 1
          ) node.data_indices;
          !sum /. float_of_int !count
      | (Some var, Some split) ->
          if z.(var) <= split then
            traverse (Option.get node.left)
          else
            traverse (Option.get node.right)
      | _ -> failwith "Invalid tree state"
    in
    traverse tree.root

  let predict forest x z =
    let predictions = Array.map (fun tree -> predict_tree tree x z) forest.trees in
    Array.fold_left (+.) 0.0 predictions /. float_of_int forest.n_trees
end

let estimate_conditional_mean data =
  let n = Array.length data in
  let x = Array.map (fun obs -> obs.x) data in
  let z = Array.map (fun obs -> obs.z) data in
  
  let forest = RandomForest.create_forest data 100 10 3 in
  Array.mapi (fun i obs -> RandomForest.predict forest obs.x obs.z) data

let estimate_nonparametric_component data theta =
  let n = Array.length data in
  let y_adj = Array.mapi (fun i obs -> obs.y -. theta *. obs.x) data in
  
  let forest = RandomForest.create_forest data 100 10 3 in
  Array.mapi (fun i obs -> RandomForest.predict forest obs.x obs.z) data

let estimate_all data model_type =
  let n = Array.length data in
  
  (* E[M_j(X)|Z] estimation *)
  let m_hat = estimate_conditional_mean data in
  
  (* f(Z) estimation *)
  let f_hat = estimate_nonparametric_component data 0.0 in
  
  (* Variance estimation *)
  let v_hat = match model_type with
  | Models.PartiallyLinear ->
      let residuals = Array.mapi (fun i obs ->
        let pred = m_hat.(i) +. f_hat.(i) in
        (obs.y -. pred) ** 2.0
      ) data in
      [| estimate_conditional_mean { data with 
           x = Array.map (fun r -> r) residuals } |]
  | Models.GeneralizedPartiallyLinear g ->
      let residuals = Array.mapi (fun i obs ->
        let pred = g(m_hat.(i) +. f_hat.(i)) in
        (obs.y -. pred) ** 2.0
      ) data in
      [| estimate_conditional_mean { data with 
           x = Array.map (fun r -> r) residuals } |]
  in
  
  (* h_0(Z) estimation *)
  let h_hat = Array.mapi (fun i _ ->
    let v_inv = 1.0 /. v_hat.(0).(i) in
    let num = m_hat.(i) *. v_inv in
    let den = v_inv in
    num /. den
  ) data in
  
  { m_hat; f_hat; h_hat; v_hat }

let check_rates nuisance1 nuisance2 n =
  let m_error = Array.map2 (fun m1 m2 -> (m1 -. m2) ** 2.0) 
                  nuisance1.m_hat nuisance2.m_hat
              |> Array.fold_left (+.) 0.0 
              |> sqrt in
  let f_error = Array.map2 (fun f1 f2 -> (f1 -. f2) ** 2.0)
                  nuisance1.f_hat nuisance2.f_hat
              |> Array.fold_left (+.) 0.0
              |> sqrt in
  m_error *. f_error < 1.0 /. sqrt(float_of_int n)