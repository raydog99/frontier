open Torch

let run_single_path config =
  let model = Model.create config.SimulationConfig.x0 config.s0 config.market_impact config.transaction_costs in
  let vf = ValueFunction.create () in
  ValueFunction.initialize vf config.params config.model_type;
  
  let results = ref [] in
  for _ = 1 to config.num_steps do
    let action = Strategy.execute config.strategy model config.params vf config.dt config.model_type in
    Model.apply_action model action;
    Model.step model config.params config.dt;
    ValueFunction.update vf config.params config.dt config.model_type;
    results := (Model.get_pnl model, Model.get_risk model, Model.get_entropy model) :: !results
  done;
  List.rev !results

let run_multiple_paths config =
  List.init config.num_paths (fun _ -> run_single_path config)

let calculate_statistics results =
  let num_paths = List.length results in
  let path_length = List.length (List.hd results) in
  let sum_list = List.init path_length (fun _ -> (0., 0., 0.)) in
  let sums = List.fold_left (fun acc path ->
    List.map2 (fun (sum_pnl, sum_risk, sum_entropy) (pnl, risk, entropy) ->
      (sum_pnl +. pnl, sum_risk +. risk, sum_entropy +. entropy)
    ) acc path
  ) sum_list results in
  let averages = List.map (fun (sum_pnl, sum_risk, sum_entropy) ->
    (sum_pnl /. float_of_int num_paths,
     sum_risk /. float_of_int num_paths,
     sum_entropy /. float_of_int num_paths)
  ) sums in
  
  let squared_diff_sums = List.fold_left (fun acc path ->
    List.map2 (fun (sum_sq_pnl, sum_sq_risk, sum_sq_entropy) (pnl, risk, entropy) ->
      let (avg_pnl, avg_risk, avg_entropy) = List.nth averages (List.length acc) in
      (sum_sq_pnl +. (pnl -. avg_pnl) ** 2.,
       sum_sq_risk +. (risk -. avg_risk) ** 2.,
       sum_sq_entropy +. (entropy -. avg_entropy) ** 2.)
    ) acc path
  ) sum_list results in
  
  let std_devs = List.map (fun (sum_sq_pnl, sum_sq_risk, sum_sq_entropy) ->
    (sqrt (sum_sq_pnl /. float_of_int num_paths),
     sqrt (sum_sq_risk /. float_of_int num_paths),
     sqrt (sum_sq_entropy /. float_of_int num_paths))
  ) squared_diff_sums in
  
  (averages, std_devs)