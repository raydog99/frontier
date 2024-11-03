open Torch

module PotentialFunction = struct
  type t = {
    vae: VAE.t;
    sigma_p: float;
  }

  let create ~vae ~sigma_p = { vae; sigma_p }

  let compute_gradient t x =
    let open Tensor in
    let recon, _, _ = VAE.forward t.vae x in
    sub x recon |> div_scalar t.sigma_p

  let compute_energy t x =
    let open Tensor in
    let recon, _, _ = VAE.forward t.vae x in
    let diff = sub x recon in
    let energy = 
      mul diff diff
      |> sum ~dim:[1; 2]
      |> div_scalar (2. *. t.sigma_p) in
    neg energy
end

module RaoBlackwellizedSampler = struct
  type t = {
    flow_model: FlowNetwork.t;
    potential: PotentialFunction.t;
    config: model_config;
  }

  let create ~flow_model ~vae ~config =
    let potential = PotentialFunction.create 
      ~vae ~sigma_p:config.potential_coeff in
    { flow_model; potential; config }

  let step t ~state ~conditional ~time =
    let open Tensor in
    let dt = t.config.terminal_time /. float_of_int t.config.euler_steps in
    
    (* Base flow update *)
    let input = cat [conditional; state] ~dim:1 in
    let velocity_base = FlowNetwork.forward t.flow_model input in
    let state_flow = add state (mul_scalar velocity_base dt) in
    
    (* Potential function update *)
    let velocity_potential = PotentialFunction.compute_gradient t.potential state_flow in
    add state_flow (mul_scalar velocity_potential dt)

  let generate_trajectories t ~conditional ~n_trajectories =
    let batch_size = Tensor.shape conditional |> List.hd in
    List.init n_trajectories (fun _ ->
      let initial_noise = 
        Tensor.randn [batch_size; conditional.dimensions; conditional.sequence_length]
        |> Tensor.mul_scalar t.config.initial_noise_std in
        
      let rec euler_step time state =
        if time >= t.config.terminal_time then state
        else
          let next_state = step t ~state ~conditional ~time in
          euler_step (time +. t.config.terminal_time /. float_of_int t.config.euler_steps) next_state
      in
      euler_step 0. initial_noise)
end

module Imputer = struct
  type t = {
    flow: FlowNetwork.t;
    sampler: RaoBlackwellizedSampler.t option;
    config: model_config;
  }

  let create ~flow ~vae ~config =
    let sampler = match vae with
      | Some vae -> Some (RaoBlackwellizedSampler.create ~flow_model:flow ~vae ~config)
      | None -> None
    in
    { flow; sampler; config }

  let impute t ~time_series ~n_trajectories =
    let open Tensor in
    let conditional, target, cond_mask, target_mask = 
      TimeSeries.split_condition_target time_series ~target_mask:time_series.mask in
    
    let trajectories = match t.sampler with
    | Some sampler -> 
        RaoBlackwellizedSampler.generate_trajectories sampler 
          ~conditional ~n_trajectories
    | None ->
        let generate_one () =
          let noise = randn [1; time_series.dimensions; time_series.sequence_length]
            |> mul_scalar t.config.initial_noise_std in
          
          let rec euler_step time state =
            if time >= t.config.terminal_time then state
            else
              let dt = t.config.terminal_time /. float_of_int t.config.euler_steps in
              let input = cat [conditional; state] ~dim:1 in
              let velocity = FlowNetwork.forward t.flow input in
              let next_state = add state (mul_scalar velocity dt) in
              euler_step (time +. dt) next_state
          in
          euler_step 0. noise
        in
        List.init n_trajectories generate_one
    in
    
    (* Average trajectories and combine with conditional data *)
    let samples = stack trajectories ~dim:0 in
    let mean_trajectory = mean samples ~dim:[0] |> squeeze ~dim:[0] in
    add (mul mean_trajectory target_mask) (mul conditional cond_mask)
end