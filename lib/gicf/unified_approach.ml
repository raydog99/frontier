open Torch

let decompose_loglikelihood y sigma i =
  let n = float_of_int (Tensor.shape y).(0) in
  let s = Tensor.mm (Tensor.transpose y ~dim0:0 ~dim1:1) y |> 
          Tensor.div_scalar n in
  
  match Estimation.compute_conditional_components sigma i (Tensor.shape sigma).(0) with
  | Error _ -> (neg_infinity, neg_infinity, 1.0, Tensor.zeros [|(Tensor.shape y).(0); (Tensor.shape y).(1)|])
  | Ok (sigma_ii, sigma_i_i, sigma_rest, inv_sigma_rest) ->
      (* Compute marginal likelihood *)
      let marginal_ll = match Numerical.safe_logdet sigma_rest with
        | Error _ -> neg_infinity
        | Ok logdet ->
            let trace_term = Tensor.mm inv_sigma_rest (Tensor.mm (Tensor.transpose sigma_i_i ~dim0:0 ~dim1:1) sigma_i_i) |>
                            Tensor.trace |> Tensor.item in
            -.(logdet) -. trace_term
      in
      
      (* Compute conditional components *)
      let z_i = Tensor.mm y inv_sigma_rest in
      let tau_i = Tensor.get sigma_ii 0 0 |> Tensor.item in
      let conditional_ll = -.(log tau_i) -. 
        (Tensor.dot (Tensor.mm z_i sigma_i_i) (Tensor.mm z_i sigma_i_i) |> 
         Tensor.item) /. (n *. tau_i) in
         
      (marginal_ll, conditional_ll, tau_i, z_i)

let compute_penalties sigma tau_values gamma =
  try
    let p = (Tensor.shape sigma).(0) in
    
    (* Off-diagonal penalty *)
    let off_diag = Tensor.sub sigma (Tensor.diagonal sigma ~dim1:0 ~dim2:1) in
    let off_diag_penalty = Tensor.sum (Tensor.abs off_diag) |> 
                          Tensor.mul_scalar gamma |> Tensor.item in
    
    (* Tau penalty *)
    let tau_penalty = Array.fold_left (fun acc tau ->
      acc +. (1.0 /. tau)
    ) 0.0 tau_values in
    
    (* Verify *)
    match Numerical.safe_inverse sigma with
    | Error _ -> Error "Failed to compute inverse for penalty verification"
    | Ok inv_sigma ->
        let diag_inv = Tensor.diagonal inv_sigma ~dim1:0 ~dim2:1 in
        let valid = Array.mapi (fun i tau ->
          let theta_ii = Tensor.get diag_inv i |> Tensor.item in
          Float.abs (1.0 /. tau -. theta_ii) < 1e-10
        ) tau_values |> Array.for_all (fun x -> x) in
        
        if valid then
          Ok { off_diagonal = off_diag_penalty; 
               tau = tau_penalty; 
               total = off_diag_penalty +. tau_penalty }
        else
          Error "Not satisfied"
  with _ ->
    Error "Failed to compute penalties"

let regression_component y z_i tau_i gamma_i gamma =
  let n = float_of_int (Tensor.shape y).(0) in
  
  (* Compute residuals *)
  let residuals = Tensor.sub y (Tensor.mm z_i gamma_i) in
  let squared_norm = Tensor.dot residuals residuals |> Tensor.item in
  
  (* Compute regression components *)
  let log_term = -.(log tau_i) in
  let residual_term = -.(squared_norm /. (n *. tau_i)) in
  let lasso_term = -.(2.0 *. gamma *. (Tensor.norm gamma_i ~p:1 |> Tensor.item)) in
  let tau_penalty = -.(1.0 /. tau_i) in
  
  (log_term +. residual_term +. lasso_term +. tau_penalty, 
   squared_norm /. (n -. float_of_int (Tensor.shape gamma_i).(0)))

let estimate ?(max_iter=100) ?(tol=1e-6) method_type y params =
  let n = (Tensor.shape y).(0) in
  let p = (Tensor.shape y).(1) in
  
  (* Select appropriate initialization based on method *)
  let init_sigma = match method_type with
    | MLE | MLEWithGraph -> 
        Tensor.diagonal (Tensor.mm (Tensor.transpose y ~dim0:0 ~dim1:1) y |> 
                       Tensor.div_scalar (float_of_int n)) ~dim1:0 ~dim2:1
    | RidgeRegularized | RidgeWithGraph -> 
        Estimation.initialize_sigma y params
    | Covglasso | CovglassoWithGraph | RidgeCovglasso | RidgeCovglassoWithGraph ->
        let s = Tensor.mm (Tensor.transpose y ~dim0:0 ~dim1:1) y |> 
                Tensor.div_scalar (float_of_int n) in
        Numerical.stabilize_matrix s 1e-10
  in
  
  let sigma = ref init_sigma in
  let tau_values = Array.make p 1.0 in
  let prev_obj = ref neg_infinity in
  
  try
    for iter = 1 to max_iter do
      let curr_sigma = Numerical.stabilize_matrix !sigma 1e-10 in
      
      (* Update each variable *)
      for i = 0 to p-1 do
        let (marginal_ll, conditional_ll, tau_i, z_i) = 
          decompose_loglikelihood y curr_sigma i in
          
        tau_values.(i) <- tau_i;
        
        (* Get boundary based on method and graph *)
        let bd_i = match method_type with
          | MLEWithGraph | RidgeWithGraph | CovglassoWithGraph | RidgeCovglassoWithGraph ->
              begin match params.graph with
                | Some g -> List.filter (fun j -> 
                    List.mem (i,j) g.edges || List.mem (j,i) g.edges
                  ) (List.init p (fun x -> if x <> i then [x] else []) |> List.flatten)
                | None -> []
              end
          | _ -> List.init p (fun x -> if x <> i then x else -1) |> 
                List.filter (fun x -> x >= 0)
        in
        
        (* Update coefficients *)
        let gamma_i = Tensor.index_select curr_sigma ~dim:0 ~index:(Tensor.of_int1 [i]) in
        let (new_gamma, new_tau) = match method_type with
          | MLE | MLEWithGraph -> 
              (gamma_i, tau_i)
          | RidgeRegularized | RidgeWithGraph ->
              let components = Estimation.compute_conditional_components curr_sigma i p in
              begin match components with
                | Error _ -> (gamma_i, tau_i)
                | Ok _ -> 
                    let new_tau = Estimation.estimate_tau y z_i gamma_i tau_i in
                    (gamma_i, new_tau)
              end
          | _ ->
              let new_tau = Estimation.estimate_tau y z_i gamma_i tau_i in
              let new_gamma = List.fold_left (fun acc j ->
                let b_ij = Tensor.get z_i j |> Tensor.item in
                let c_ij = Tensor.dot z_i z_i |> Tensor.item in
                let updated_coef = Estimation.update_coef 
                  (Tensor.float_scalar b_ij) 
                  (Tensor.float_scalar c_ij) 
                  params.gamma new_tau in
                Tensor.copy_update acc 0 j updated_coef
              ) gamma_i bd_i in
              (new_gamma, new_tau)
        in
        
        (* Update sigma *)
        sigma := Estimation.update_sigma !sigma i new_tau new_gamma
      done;
      
      (* Check convergence *)
      let curr_obj = Estimation.compute_objective !sigma y params in
      if Float.abs (curr_obj -. !prev_obj) < tol then
        raise Exit;
        
      prev_obj := curr_obj
    done;
    Error "Maximum iterations reached"
  with 
    | Exit -> Ok !sigma
    | e -> Error (Printexc.to_string e)