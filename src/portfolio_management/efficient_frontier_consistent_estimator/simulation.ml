open Torch

type scenario =
  | Normal
  | StudentT of float
  | CCCGARCH

let generate_eigenvalues p =
  let small = Tensor.full [p / 5] 0.5 in
  let medium = Tensor.full [2 * p / 5] 1.0 in
  let large = Tensor.full [2 * p / 5] 5.0 in
  Tensor.cat [small; medium; large] 0

let generate_data n p scenario =
  let mean_vector = Tensor.uniform [p; 1] ~from:(-0.2) ~to_:0.2 in
  let eigenvalues = generate_eigenvalues p in
  let q = Tensor.qr (Tensor.randn [p; p]) |> fst in
  let covariance_matrix = Tensor.(mm (mm q (diag eigenvalues)) (transpose2 q)) in
  
  let x = match scenario with
  | Normal -> Tensor.randn [n; p]
  | StudentT df ->
      let chi_square = Tensor.distributions.Chi2 df in
      let z = Tensor.randn [n; p] in
      Tensor.(z / sqrt (chi_square / df))
  | CCCGARCH ->
      let alpha = Tensor.uniform [p; 1] ~from:0.0 ~to_:0.1 in
      let beta = Tensor.uniform [p; 1] ~from:0.8 ~to_:0.89 in
      let omega = Tensor.((1.0 - alpha - beta) * covariance_matrix) in
      
      let rec simulate_garch t acc =
        if t = n then Tensor.stack (List.rev acc) 0
        else
          let prev_h = if t = 0 then Tensor.ones [p; 1] else List.hd acc in
          let z = Tensor.randn [p; 1] in
          let h = Tensor.(omega + alpha * (prev_h * z ** 2.0) + beta * prev_h) in
          let x = Tensor.(sqrt h * z) in
          simulate_garch (t + 1) (x :: acc)
      in
      simulate_garch 0 []
  in
  
  let y = Tensor.(mm (sqrt covariance_matrix) (transpose2 x)) in
  EfficientFrontier.create (Tensor.transpose2 y) covariance_matrix

let run_simulation n p scenario n_simulations =
  let results = Array.init n_simulations (fun _ ->
    let ef = generate_data n p scenario in
    let true_params = EfficientFrontier.estimate_parameters ef Estimators.Sample in
    
    List.map (fun (estimator_name, estimator) ->
      let (r_est, v_est, s_est) = EfficientFrontier.estimate_parameters ef estimator in
      let (r_true, v_true, s_true) = true_params in
      let loss_r = EfficientFrontier.quadratic_loss r_true r_est in
      let loss_v = EfficientFrontier.quadratic_loss v_true v_est in
      let loss_s = EfficientFrontier.quadratic_loss s_true s_est in
      (estimator_name,
       Tensor.to_float0_exn r_est, Tensor.to_float0_exn v_est, Tensor.to_float0_exn s_est,
       Tensor.to_float0_exn loss_r, Tensor.to_float0_exn loss_v, Tensor.to_float0_exn loss_s)
    ) Estimators.all
  ) in
  
  let process_results results =
    List.map (fun (estimator_name, rs, vs, ss, lrs, lvs, lss) ->
      let ci_r = Statistics.confidence_interval rs 0.05 in
      let ci_v = Statistics.confidence_interval vs 0.05 in
      let ci_s = Statistics.confidence_interval ss 0.05 in
      (estimator_name, 
       Statistics.mean rs, ci_r,
       Statistics.mean vs, ci_v,
       Statistics.mean ss, ci_s,
       Statistics.mean lrs,
       Statistics.mean lvs,
       Statistics.mean lss)
    ) results
  in

  let results = Array.to_list results in
  let grouped_results = 
    List.map (fun (name, _, _, _, _, _, _) -> 
      (name, List.map (fun sim -> List.assoc name sim) results)
    ) Estimators.all
  in
  process_results grouped_results

let compare_estimators results =
  let base_estimator = "Consistent" in
  let base_results = List.assoc base_estimator results in
  List.filter (fun (name, _) -> name <> base_estimator) results
  |> List.map (fun (name, (_, rs, _, vs, _, ss, _, _, _)) ->
    let (_, base_rs, _, base_vs, _, base_ss, _, _, _) = base_results in
    let p_value_r = Statistics.t_test rs base_rs in
    let p_value_v = Statistics.t_test vs base_vs in
    let p_value_s = Statistics.t_test ss base_ss in
    (name, p_value_r, p_value_v, p_value_s)
  )