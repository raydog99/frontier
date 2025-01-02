open Torch
open Lwt.Infix

type problem_type = MeanVariance | RiskMinimization
type constraint_type = NoConstraint | BoxConstraint of float * float | GroupConstraint of int list list * float list * float list

type t = {
  n : int;
  returns : float tensor;
  covariance : float tensor;
  risk_aversion : float;
  cardinality : float;
  problem_type : problem_type;
  baseline_portfolio : float tensor option;
  max_community_size : int option;
  constraint : constraint_type;
}

let create n returns covariance risk_aversion cardinality problem_type ?max_community_size ?constraint () =
  if n <= 0 then invalid_arg "Number of assets must be positive";
  if Tensor.shape returns <> [n] then invalid_arg "Returns tensor shape mismatch";
  if Tensor.shape covariance <> [n; n] then invalid_arg "Covariance matrix shape mismatch";
  if risk_aversion <= 0. then invalid_arg "Risk aversion must be positive";
  if cardinality <= 0. || cardinality > 1. then invalid_arg "Cardinality must be between 0 and 1";
  let baseline_portfolio = 
    match problem_type with
    | RiskMinimization -> Some (Tensor.ones [n] |> Tensor.div_scalar (float n))
    | _ -> None
  in
  let constraint = match constraint with
  | Some c -> c
  | None -> NoConstraint
  in
  { n; returns; covariance; risk_aversion; cardinality; problem_type; baseline_portfolio; max_community_size; constraint }

let correlation_from_covariance covariance =
  let std_dev = Tensor.sqrt (Tensor.diag covariance) in
  let inv_std_dev = Tensor.reciprocal std_dev in
  Tensor.mm (Tensor.mm (Tensor.diag inv_std_dev) covariance) (Tensor.diag inv_std_dev)

let preprocess_correlation_matrix t =
  correlation_from_covariance t.covariance

let fit_marchenko_pastur t =
  let corr = preprocess_correlation_matrix t in
  let eigenvalues = Tensor.symeig ~eigenvectors:false corr in
  let lambda_plus = Tensor.max eigenvalues |> Tensor.float_value in
  let lambda_minus = Tensor.min eigenvalues |> Tensor.float_value in
  let q = Float.of_int t.n /. (Tensor.size t.returns 1 |> Float.of_int) in
  let lambda_plus_theoretical = (1. +. sqrt q) ** 2. in
  let lambda_minus_theoretical = (1. -. sqrt q) ** 2. in
  (lambda_minus, lambda_plus)

let clean_correlation_matrix t =
  let corr = preprocess_correlation_matrix t in
  let eigenvalues, eigenvectors = Tensor.symeig ~eigenvectors:true corr in
  let lambda_minus, lambda_plus = fit_marchenko_pastur t in
  let mask = Tensor.(eigenvalues > (float lambda_plus)) in
  let cleaned_eigenvalues = Tensor.(eigenvalues * mask) in
  Tensor.mm (Tensor.mm eigenvectors (Tensor.diag cleaned_eigenvalues)) (Tensor.transpose eigenvectors 0 1)

let modularity_matrix c_star =
  let k = Tensor.sum c_star 1 in
  let m = Tensor.sum k |> Tensor.float_value in
  let expected = Tensor.outer_product k k |> Tensor.div_scalar m in
  Tensor.(c_star - expected)

let cluster t =
  let c_star = clean_correlation_matrix t in
  let rec newman_clustering communities =
    if Tensor.size communities 0 = t.n then communities
    else
      let b = modularity_matrix c_star in
      let eigenvalues, eigenvectors = Tensor.symeig ~eigenvectors:true b in
      let leading_eigenvector = Tensor.select eigenvectors 1 (-1) in
      let new_communities = Tensor.sign leading_eigenvector in
      let updated_communities = Tensor.cat [communities; new_communities] 0 in
      match t.max_community_size with
      | Some max_size ->
          let community_sizes = Tensor.sum updated_communities 1 in
          if Tensor.any (Tensor.gt community_sizes (Tensor.of_int1 max_size)) then
            let largest_community_index = Tensor.argmax community_sizes 0 |> Tensor.int_value in
            let large_community = Tensor.select updated_communities largest_community_index 0 in
            let sub_c_star = Tensor.masked_select c_star (Tensor.outer_product large_community large_community) in
            let sub_c_star = Tensor.reshape sub_c_star [Tensor.sum large_community; Tensor.sum large_community] in
            let sub_communities = newman_clustering (Tensor.empty [0; Tensor.sum large_community]) in
            let new_updated_communities = Tensor.cat 
              [Tensor.slice updated_communities 0 0 largest_community_index;
               Tensor.slice updated_communities 0 (largest_community_index + 1) (-1);
               sub_communities] 0
            in
            newman_clustering new_updated_communities
          else
            updated_communities
      | None -> updated_communities
  in
  newman_clustering (Tensor.empty [0; t.n])

let risk_rebalance t communities =
  let n_communities = Tensor.size communities 0 in
  let community_returns = List.init n_communities (fun i ->
    let community = Tensor.select communities 0 i in
    let mask = Tensor.(community == (ones_like community)) in
    Tensor.masked_select t.returns mask
  ) in
  let community_covariances = List.init n_communities (fun i ->
    let community = Tensor.select communities 0 i in
    let mask = Tensor.(community == (ones_like community)) in
    let sub_covariance = Tensor.masked_select t.covariance (Tensor.outer_product mask mask) in
    Tensor.reshape sub_covariance [Tensor.sum mask; Tensor.sum mask]
  ) in
  let sum_returns_norm = List.fold_left (fun acc r -> acc +. Tensor.norm r |> Tensor.float_value) 0. community_returns in
  let sum_covariance_norm = List.fold_left (fun acc c -> acc +. Tensor.norm c |> Tensor.float_value) 0. community_covariances in
  t.risk_aversion *. (sum_returns_norm /. sum_covariance_norm)

let apply_constraint t weights =
  match t.constraint with
  | NoConstraint -> weights
  | BoxConstraint (lower, upper) ->
      Tensor.clamp weights ~min:lower ~max:upper
  | GroupConstraint (groups, lower_bounds, upper_bounds) ->
      let constrained_weights = Tensor.clone weights in
      List.iteri (fun i group ->
        let group_weights = List.map (fun idx -> Tensor.get constrained_weights idx) group in
        let group_sum = List.fold_left (+.) 0. group_weights in
        let lower = List.nth lower_bounds i in
        let upper = List.nth upper_bounds i in
        let scale = 
          if group_sum < lower then lower /. group_sum
          else if group_sum > upper then upper /. group_sum
          else 1.
        in
        List.iter (fun idx -> 
          Tensor.set constrained_weights idx (Tensor.get constrained_weights idx *. scale)
        ) group
      ) groups;
      constrained_weights

let optimize_subproblem t community rebalanced_risk_aversion =
  Lwt.return (
    try
      let mask = Tensor.(community == (ones_like community)) in
      let sub_returns = Tensor.masked_select t.returns mask in
      let sub_covariance = Tensor.masked_select t.covariance (Tensor.outer_product mask mask) in
      let sub_covariance = Tensor.reshape sub_covariance [Tensor.sum mask; Tensor.sum mask] in
      
      let n_sub = Tensor.sum mask |> Tensor.int_value in
      let result = match t.problem_type with
      | MeanVariance ->
          let h = Tensor.(rebalanced_risk_aversion * sub_covariance) in
          let f = Tensor.neg sub_returns in
          let a = Tensor.ones [n_sub] in
          let b = Tensor.full [1] (t.cardinality *. (float n_sub /. Float.of_int t.n)) in
          
          let rho = 1.0 in
          let num_iterations = 1000 in
          let rec admm_iter x z u iter =
            if iter = 0 then x
            else
              let x_new = Tensor.(matmul (inverse (h + (eye n_sub []) * float rho)) (f - (mm (transpose a 0 1) z) + (mm (transpose a 0 1) u) - (u * float rho))) in
              let z_new = Tensor.max (Tensor.min (x_new + u) (Tensor.ones [n_sub])) (Tensor.zeros [n_sub]) in
              let u_new = Tensor.(u + x_new - z_new) in
              admm_iter x_new z_new u_new (iter - 1)
          in
          let initial_x = Tensor.(ones [n_sub] / float n_sub) in
          let initial_z = Tensor.zeros [n_sub] in
          let initial_u = Tensor.zeros [n_sub] in
          admm_iter initial_x initial_z initial_u num_iterations

      | RiskMinimization ->
          let sub_baseline = Tensor.masked_select (Option.get t.baseline_portfolio) mask in
          let h = sub_covariance in
          let f = Tensor.zeros [n_sub] in
          let a = Tensor.(mm sub_covariance sub_baseline) in
          let b = Tensor.full [1] (rebalanced_risk_aversion *. (Tensor.dot sub_baseline (Tensor.mv sub_covariance sub_baseline) |> Tensor.float_value)) in
          
          let rho = 1.0 in
          let num_iterations = 1000 in
          let rec admm_iter x z u iter =
            if iter = 0 then x
            else
              let x_new = Tensor.(matmul (inverse (h + (eye n_sub []) * float rho)) (f - (mm (transpose a 0 1) z) + (mm (transpose a 0 1) u) - (u * float rho))) in
              let z_new = Tensor.(x_new + u - (a * ((dot (x_new + u) a - b) / (dot a a)))) in
              let u_new = Tensor.(u + x_new - z_new) in
              admm_iter x_new z_new u_new (iter - 1)
          in
          let initial_x = sub_baseline in
          let initial_z = Tensor.zeros [n_sub] in
          let initial_u = Tensor.zeros [n_sub] in
          admm_iter initial_x initial_z initial_u num_iterations
      in
      Ok (apply_constraint t result)
    with
    | _ -> Error "Subproblem optimization failed"
  )

let aggregate_results t communities subproblem_solutions =
  let n_communities = List.length subproblem_solutions in
  let full_solution = Tensor.zeros [t.n] in
  List.iteri (fun i sol ->
    let community = Tensor.select communities 0 i in
    let mask = Tensor.(community == (ones_like community)) in
    Tensor.masked_scatter_ full_solution mask sol
  ) subproblem_solutions;
  full_solution

let optimize t =
  Lwt.try_bind
    (fun () ->
      let communities = cluster t in
      let rebalanced_risk_aversion = risk_rebalance t communities in
      let n_communities = Tensor.size communities 0 in
      Lwt_list.map_p
        (fun i ->
          let community = Tensor.select communities 0 i in
          optimize_subproblem t community rebalanced_risk_aversion
        )
        (List.init n_communities (fun i -> i))
    )
    (fun subproblem_results ->
      let subproblem_solutions = 
        List.map
          (function
            | Ok solution -> solution
            | Error msg -> failwith msg)
          subproblem_results
      in
      Lwt.return (Ok (aggregate_results t communities subproblem_solutions))
    )
    (fun exn ->
      Lwt.return (Error (Printexc.to_string exn))
    )

let portfolio_return t weights =
  Tensor.dot t.returns weights |> Tensor.float_value

let portfolio_risk t weights =
  Tensor.(sqrt (dot weights (mv t.covariance weights))) |> Tensor.float_value

let sharpe_ratio t weights =
  let ret = portfolio_return t weights in
  let risk = portfolio_risk t weights in
  ret /. risk

let get_communities t communities =
  let n_communities = Tensor.size communities 0 in
  List.init n_communities (fun i ->
    let community = Tensor.select communities 0 i in
    let mask = Tensor.(community == (ones_like community)) in
    let community_assets = Tensor.masked_select t.returns mask in
    (i, community_assets)
  )

let get_community_statistics t communities =
  let n_communities = Tensor.size communities 0 in
  List.init n_communities (fun i ->
    let community = Tensor.select communities 0 i in
    let mask = Tensor.(community == (ones_like community)) in
    let community_returns = Tensor.masked_select t.returns mask in
    let community_covariance = Tensor.masked_select t.covariance (Tensor.outer_product mask mask) in
    let community_covariance = Tensor.reshape community_covariance [Tensor.sum mask; Tensor.sum mask] in
    let size = Tensor.sum mask |> Tensor.int_value in
    let avg_return = Tensor.mean community_returns |> Tensor.float_value in
    let avg_risk = Tensor.mean (Tensor.diag community_covariance) |> Tensor.float_value |> sqrt in
    let avg_correlation = 
      let corr_matrix = correlation_from_covariance community_covariance in
      (Tensor.sum corr_matrix |> Tensor.float_value -. float size) /. (float (size * (size - 1)))
    in
    (i, avg_return, avg_risk, avg_correlation)
  )

let test t =
  let test_optimization () =
    optimize t >>= function
    | Ok weights ->
        let ret = portfolio_return t weights in
        let risk = portfolio_risk t weights in
        let sr = sharpe_ratio t weights in
        Lwt_io.printf "Test optimization:\nReturn: %f\nRisk: %f\nSharpe Ratio: %f\n" ret risk sr
    | Error msg ->
        Lwt_io.printf "Test optimization failed: %s\n" msg
  in
  let test_clustering () =
    let communities = cluster t in
    let stats = get_community_statistics t communities in
    Lwt_list.iter_s (fun (i, avg_return, avg_risk, avg_correlation) ->
      Lwt_io.printf "Community %d: Avg Return: %f, Avg Risk: %f, Avg Correlation: %f\n" 
        i avg_return avg_risk avg_correlation
    ) stats
  in
  
  let test_constraints () =
    let random_weights = Tensor.rand [t.n] in
    let constrained_weights = apply_constraint t random_weights in
    Lwt_io.printf "Original weights sum: %f\nConstrained weights sum: %f\n"
      (Tensor.sum random_weights |> Tensor.float_value)
      (Tensor.sum constrained_weights |> Tensor.float_value)
  in
  
  Lwt.join [
    test_optimization ();
    test_clustering ();
    test_constraints ();
  ]