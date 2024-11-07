open Types
open Torch

let soft_threshold x gamma =
  let sign = Tensor.sign x in
  let abs_x = Tensor.abs x in
  let shifted = Tensor.sub abs_x (Tensor.float_vec [|gamma|]) in
  let thresholded = Tensor.relu shifted in
  Tensor.mul sign thresholded

let ridge_regularize s lambda =
  let n = (Tensor.shape s).(0) in
  let identity = Tensor.eye n in
  Tensor.add s (Tensor.mul_scalar identity lambda)

let estimate_tau y z_bd gamma_bd tau =
  let n = float_of_int ((Tensor.shape y).(0)) in
  let residuals = Tensor.sub y (Tensor.mm z_bd gamma_bd) in
  let squared_norm = Tensor.dot residuals residuals in
  squared_norm /. n

let compute_conditional_components sigma i p =
  let i_indices = List.filter (fun j -> j <> i) (List.init p (fun x -> x)) in
  
  let sigma_ii = Tensor.slice sigma ~dim:0 ~start:i ~length:1 ~step:1 |>
                 Tensor.slice ~dim:1 ~start:i ~length:1 ~step:1 in
                 
  let sigma_i = Tensor.index_select sigma ~dim:0 ~index:(Tensor.of_int1 i_indices) in
  let sigma_i_i = Tensor.index_select sigma_i ~dim:1 ~index:(Tensor.of_int1 [i]) in
  let sigma_i_rest = Tensor.index_select sigma_i ~dim:1 ~index:(Tensor.of_int1 i_indices) in
  
  match Numerical.safe_inverse sigma_i_rest with
  | Error msg -> Error msg
  | Ok inv_sigma_rest ->
      Ok (sigma_ii, sigma_i_i, sigma_i_rest, inv_sigma_rest)

let compute_pseudo_predictors y sigma_components =
  match sigma_components with
  | Ok (_, _, _, inv_sigma_rest) -> Tensor.mm y inv_sigma_rest
  | Error _ -> Tensor.zeros [|(Tensor.shape y).(0); (Tensor.shape y).(1)|]

let update_coef b_ij c_ij gamma tau =
  let threshold = soft_threshold b_ij gamma in
  Tensor.div threshold c_ij

let compute_objective sigma y params =
  let n = float_of_int (Tensor.shape y).(0) in
  let s = Tensor.mm (Tensor.transpose y ~dim0:0 ~dim1:1) y |> 
          Tensor.div_scalar (float_of_int n) in
          
  let s_ridge = ridge_regularize s params.lambda in
  
  match Numerical.safe_logdet sigma, Numerical.safe_inverse sigma with
  | Ok logdet, Ok inv_sigma ->
      let trace_term = Tensor.mm inv_sigma s_ridge |> Tensor.trace |> Tensor.item in
      let ll = -2.0 *. logdet -. trace_term in
      
      let off_diag = Tensor.sub sigma (Tensor.diagonal sigma ~dim1:0 ~dim2:1) in
      let penalty = Tensor.sum (Tensor.abs off_diag) |> 
                   Tensor.mul_scalar params.gamma |> Tensor.item in
                   
      ll -. penalty
  | Error _, _ | _, Error _ -> neg_infinity

let initialize_sigma y params =
  let n = (Tensor.shape y).(0) in
  let s = Tensor.mm (Tensor.transpose y ~dim0:0 ~dim1:1) y |> 
          Tensor.div_scalar (float_of_int n) in
  let s_ridge = ridge_regularize s params.lambda in
  Tensor.diagonal s_ridge ~dim1:0 ~dim2:1

let update_sigma sigma i tau gamma =
  let p = (Tensor.shape sigma).(0) in
  let new_sigma = Tensor.copy sigma in
  
  (* Update row and column i *)
  for j = 0 to p-1 do
    if i <> j then begin
      let gamma_j = Tensor.get gamma j |> Tensor.item in
      Tensor.copy_update new_sigma i j (Tensor.float_scalar gamma_j);
      Tensor.copy_update new_sigma j i (Tensor.float_scalar gamma_j)
    end
  done;
  
  (* Update diagonal element *)
  Tensor.copy_update new_sigma i i (Tensor.float_scalar tau);
  new_sigma

let fit ?(max_iter=100) ?(tol=1e-6) params y =
  let n = (Tensor.shape y).(0) in
  let p = (Tensor.shape y).(1) in
  
  let sigma = ref (initialize_sigma y params) in
  let prev_obj = ref neg_infinity in
  let stats = ref {
    iterations = 0;
    final_delta = 0.0;
    objective_values = [||];
    condition_numbers = [||];
    elapsed_time = 0.0;
  } in
  
  let start_time = Unix.gettimeofday() in
  let converged = ref false in
  
  while not !converged && !stats.iterations < max_iter do
    let curr_sigma = Numerical.stabilize_matrix !sigma 1e-10 in
    
    (* Update each variable *)
    for i = 0 to p-1 do
      let components = compute_conditional_components curr_sigma i p in
      match components with
      | Error _ -> ()
      | Ok comps ->
          let z_i = compute_pseudo_predictors y comps in
          let gamma_i = Tensor.index_select curr_sigma ~dim:0 ~index:(Tensor.of_int1 [i]) in
          let tau_i = Tensor.get curr_sigma i i |> Tensor.item in
          
          let new_tau = estimate_tau y z_i gamma_i tau_i in
          let new_gamma = update_coef z_i gamma_i params.gamma new_tau in
          
          sigma := update_sigma !sigma i new_tau new_gamma
    done;
    
    (* Update convergence statistics *)
    let curr_obj = compute_objective !sigma y params in
    stats := {
      !stats with
      iterations = !stats.iterations + 1;
      final_delta = Float.abs (curr_obj -. !prev_obj);
      objective_values = Array.append !stats.objective_values [|curr_obj|];
      condition_numbers = Array.append !stats.condition_numbers 
        [|Numerical.monitor_conditioning !sigma 1e6 |> function
           | Ok _ -> Tensor.condition !sigma |> Tensor.item
           | Error _ -> infinity|];
      elapsed_time = Unix.gettimeofday() -. start_time;
    };
    
    converged := Float.abs (curr_obj -. !prev_obj) < tol;
    prev_obj := curr_obj
  done;
  
  if !converged then Ok (!sigma, !stats)
  else Error "Failed to converge"