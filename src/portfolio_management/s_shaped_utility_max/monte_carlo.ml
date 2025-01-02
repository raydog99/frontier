open Torch

let run_wealth_process model_params initial_wealth control t_steps num_runs =
  let dt = model_params.Model.t /. float_of_int t_steps in
  let sqrt_dt = sqrt dt in
  
  let run_path () =
    let rec loop t wealth acc =
      if t >= model_params.Model.t then List.rev acc
      else
        let dw = Tensor.randn [1] in
        let control_value = control t wealth in
        let drift = Tensor.((model_params.alpha * wealth) + (control_value * wealth * model_params.theta * model_params.sigma)) in
        let diffusion = Tensor.(control_value * wealth * model_params.sigma * dw) in
        let new_wealth = Tensor.(wealth + (drift * float_vec [|dt|]) + (diffusion * float_vec [|sqrt_dt|])) in
        loop (t +. dt) new_wealth (new_wealth :: acc)
    in
    loop 0. initial_wealth [initial_wealth]
  in
  
  List.init num_runs (fun _ -> run_path ())

let compute_expected_utility utility wealth_processes =
  let module U = (val utility : Utility.Utility) in
  let terminal_values = List.map List.hd wealth_processes in
  let utilities = List.map U.evaluate terminal_values in
  Tensor.mean (Tensor.stack utilities ~dim:0)