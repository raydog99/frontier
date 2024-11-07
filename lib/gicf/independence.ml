open Types
open Torch

type test_result = {
  statistic: float;
  p_value: float;
  test_method: independence_test_method;
  sample_size: int;
  degrees_freedom: int;
  confidence_interval: float * float;
}

let compute_bivariate_likelihood x y =
  let n = float_of_int (Tensor.shape x).(0) in
  let corr = Statistics.compute_correlation x y in
  let det = 1.0 -. corr *. corr in
  -.(n /. 2.0) *. (log det)

let compute_independent_likelihood x y =
  let n = float_of_int (Tensor.shape x).(0) in
  0.0  (* Log likelihood under independence *)

let compute_mutual_information x y =
  let n = float_of_int (Tensor.shape x).(0) in
  let joint_ll = compute_bivariate_likelihood x y in
  let indep_ll = compute_independent_likelihood x y in
  (joint_ll -. indep_ll) /. n

let compute_partial_correlation x y data =
  let n = (Tensor.shape data).(0) in
  let p = (Tensor.shape data).(1) in
  
  (* Fit regressions *)
  let res_x = residualize_on_others x data in
  let res_y = residualize_on_others y data in
  
  (* Compute correlation of residuals *)
  Statistics.compute_correlation res_x res_y

let test_marginal_independence y i j method_type =
  let n = (Tensor.shape y).(0) in
  let x_i = Tensor.select y ~dim:1 ~index:i in
  let x_j = Tensor.select y ~dim:1 ~index:j in
  
  match method_type with
  | Fisher ->
      let r = Statistics.compute_correlation x_i x_j in
      let z = 0.5 *. (log ((1. +. r) /. (1. -. r))) in
      let se = 1. /. sqrt (float_of_int (n - 3)) in
      let stat = z /. se in
      let p_val = Statistics.compute_p_value_normal stat in
      let ci = (tanh (z -. 1.96 *. se), tanh (z +. 1.96 *. se)) in
      { statistic = stat; p_value = p_val; test_method = Fisher;
        sample_size = n; degrees_freedom = n-3; confidence_interval = ci }
        
  | Likelihood ->
      let ll_full = compute_bivariate_likelihood x_i x_j in
      let ll_indep = compute_independent_likelihood x_i x_j in
      let stat = 2. *. (ll_full -. ll_indep) in
      let p_val = Statistics.compute_p_value_chisq stat 1 in
      let ci = (-1.96 /. sqrt (float_of_int n), 1.96 /. sqrt (float_of_int n)) in
      { statistic = stat; p_value = p_val; test_method = Likelihood;
        sample_size = n; degrees_freedom = 1; confidence_interval = ci }
        
  | Mutual ->
      let mi = compute_mutual_information x_i x_j in
      let stat = 2. *. float_of_int n *. mi in
      let p_val = Statistics.compute_p_value_chisq stat 1 in
      let ci = (-1.96 /. sqrt (float_of_int n), 1.96 /. sqrt (float_of_int n)) in
      { statistic = stat; p_value = p_val; test_method = Mutual;
        sample_size = n; degrees_freedom = 1; confidence_interval = ci }
        
  | Partial ->
      let pc = compute_partial_correlation x_i x_j y in
      let t_stat = pc *. sqrt (float_of_int (n-2)) /. sqrt (1. -. pc *. pc) in
      let p_val = Statistics.compute_p_value_t t_stat (n-2) in
      let ci = (-1.96 /. sqrt (float_of_int (n-2)), 1.96 /. sqrt (float_of_int (n-2))) in
      { statistic = t_stat; p_value = p_val; test_method = Partial;
        sample_size = n; degrees_freedom = n-2; confidence_interval = ci }

let learn_independence_graph ?(alpha=0.05) y =
  let p = (Tensor.shape y).(1) in
  let test_methods = [Fisher; Likelihood; Mutual; Partial] in
  let edges = ref [] in
  
  for i = 0 to p-2 do
    for j = i+1 to p-1 do
      let test_results = List.map (fun method_type ->
        test_marginal_independence y i j method_type
      ) test_methods in
      
      (* Use Fisher's method to combine p-values *)
      let combined_stat = -2.0 *. List.fold_left (fun acc result ->
        acc +. log result.p_value
      ) 0.0 test_results in
      
      let combined_p_value = Statistics.compute_p_value_chisq 
        combined_stat (2 * List.length test_methods) in
      
      if combined_p_value < alpha then
        edges := (i, j) :: !edges
    done
  done;
  
  { num_vertices = p; edges = !edges }