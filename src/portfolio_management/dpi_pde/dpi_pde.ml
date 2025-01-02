open Torch

module PDE = struct
  type t = {
    dim : int;
    time_horizon : float;
    terminal_condition : Tensor.t -> Tensor.t;
    nonlinearity : Tensor.t -> Tensor.t -> Tensor.t;
  }
end

module Network = struct
  type t = {
    layers : nn;
    device : Device.t;
  }

  let create ~input_dim ~hidden_dim ~num_layers ~device =
    let rec build_layers n acc =
      if n = 0 then acc
      else
        let in_dim = if n = num_layers then input_dim else hidden_dim in
        let layer = 
          Sequential.of_list [
            Linear.create ~in_features:in_dim ~out_features:hidden_dim;
            Activation.relu;
          ]
        in
        build_layers (n-1) (layer :: acc)
    in
    let layers = 
      Sequential.of_list (
        build_layers num_layers [] @ 
        [Linear.create ~in_features:hidden_dim ~out_features:1]
      )
    in
    { layers; device }

  let forward t x =
    let x = Tensor.to_device x t.device in
    let requires_grad = Tensor.requires_grad x in
    let result = nn_forward t.layers x in
    if requires_grad then
      Tensor.set_requires_grad result true;
    result

  let gradient t x =
    let x = Tensor.to_device x t.device |> Tensor.requires_grad_ in
    let y = forward t x in
    let grad = Tensor.grad y [x] in
    match grad with
    | [g] -> g
    | _ -> failwith "Gradient computation failed"

  let create_optimized ~input_dim ~hidden_dim ~num_layers ~device =
    let init_gain = 1.0 /. Float.sqrt (float_of_int input_dim) in
    let init_weights shape =
      Tensor.randn shape ~device 
      |> Tensor.mul_scalar init_gain 
    in
    let layers = Sequential.of_list (
      build_layers num_layers [] @ 
      [Linear.create ~in_features:hidden_dim ~out_features:1 ~init_ws:init_weights]
    ) in
    { layers; device }
end

module SDE = struct
  type t = 
    | Brownian
    | GeometricBrownian of float
    | OrnsteinUhlenbeck of float

  let drift sde x =
    match sde with
    | Brownian -> Tensor.zeros_like x
    | GeometricBrownian mu -> Tensor.mul_scalar x mu
    | OrnsteinUhlenbeck theta -> Tensor.mul_scalar x (-.theta)

  let diffusion sde x =
    match sde with
    | Brownian -> Tensor.ones_like x
    | GeometricBrownian _ -> x
    | OrnsteinUhlenbeck _ -> Tensor.ones_like x

  let sample_path sde ~batch_size ~dim ~num_steps ~dt ~device =
    let dw = Tensor.randn [batch_size; num_steps; dim] ~device 
             |> Tensor.mul_scalar (Float.sqrt dt) in
    let x0 = Tensor.zeros [batch_size; dim] ~device in
    
    let rec generate_path step x_prev acc =
      if step >= num_steps then List.rev acc
      else
        let drift_term = Tensor.mul_scalar (drift sde x_prev) dt in
        let diff_term = Tensor.mul (diffusion sde x_prev) 
                                 (Tensor.select dw ~dim:1 ~index:step) in
        let x_next = Tensor.add (Tensor.add x_prev drift_term) diff_term in
        generate_path (step + 1) x_next (x_next :: acc)
    in
    let path = generate_path 0 x0 [x0] in
    Tensor.stack path ~dim:1
end

module MC = struct
  type path = {
    times : Tensor.t;
    values : Tensor.t;
    increments : Tensor.t;
  }

  let feynman_kac_estimate ~pde ~network ~x_t ~t ~dt ~num_samples ~device =
    let batch_size = Tensor.size x_t 0 in
    
    (* Generate paths *)
    let dw = Tensor.randn [batch_size; num_samples] ~device 
             |> Tensor.mul_scalar (Float.sqrt dt) in
    let path_values = Tensor.cumsum dw ~dim:1 in
    
    (* Terminal values *)
    let x_T = Tensor.add x_t (Tensor.select path_values ~dim:1 ~index:(num_samples-1)) in
    let g_T = pde.terminal_condition x_T in
    
    (* Compute integral term *)
    let rec compute_integral step acc =
      if step >= num_samples then acc
      else
        let x_s = Tensor.add x_t (Tensor.select path_values ~dim:1 ~index:step) in
        let t_s = Tensor.full [batch_size] (float_of_int step *. dt) ~device in
        let input = Tensor.cat [t_s; x_s] ~dim:1 in
        let u_s = Network.forward network input in
        let grad_u = Network.gradient network input in
        let f_s = pde.nonlinearity u_s grad_u in
        let new_acc = Tensor.add acc (Tensor.mul_scalar f_s dt) in
        compute_integral (step + 1) new_acc
    in
    let integral = compute_integral 0 (Tensor.zeros_like g_T) in
    
    Tensor.add g_T integral

  let gradient_finite_var ~pde ~network ~x_t ~t ~dt ~num_samples =
    let batch_size = Tensor.size x_t 0 in
    let device = Tensor.device x_t in
    
    (* Generate multiple paths for variance reduction *)
    let num_paths = 10 in
    let dw = Tensor.randn [batch_size; num_samples; num_paths] ~device 
             |> Tensor.mul_scalar (Float.sqrt dt) in
    
    let compute_path_gradient path_idx =
      let path_dw = Tensor.select dw ~dim:2 ~index:path_idx in
      let path_values = Tensor.cumsum path_dw ~dim:1 in
      
      (* Terminal values with control variate *)
      let x_T = Tensor.add x_t (Tensor.select path_values ~dim:1 ~index:(num_samples-1)) in
      let g_T = pde.terminal_condition x_T in
      let g_t = pde.terminal_condition x_t in
      
      let control_term = Tensor.div 
        (Tensor.sub g_T g_t)
        (Tensor.sub time_horizon (Tensor.reshape t [batch_size]))
        |> fun diff -> Tensor.mul diff path_dw in
      
      (* Function value terms *)
      let integral_term = 
        let rec compute_integral step acc =
          if step >= num_samples then acc
          else
            let x_s = Tensor.add x_t (Tensor.select path_values ~dim:1 ~index:step) in
            let t_s = Tensor.full [batch_size] (float_of_int step *. dt) ~device in
            let input = Tensor.cat [t_s; x_s] ~dim:1 in
            let f_s = pde.nonlinearity 
              (Network.forward network input)
              (Network.gradient network input) in
            let new_acc = Tensor.add acc (Tensor.mul_scalar f_s dt) in
            compute_integral (step + 1) new_acc
        in
        compute_integral 0 (Tensor.zeros_like g_T)
      in
      
      Tensor.add control_term integral_term
    in
    
    let all_grads = List.init num_paths compute_path_gradient in
    List.fold_left Tensor.add (List.hd all_grads) (List.tl all_grads)
    |> Tensor.div_scalar (float_of_int num_paths)
end

module PicardIteration = struct
  type config = {
    dt : float;
    num_steps : int;
    batch_size : int;
    device : Device.t;
  }

  let iterate ~pde ~network ~x_t ~t ~config =
    let batch_size = Tensor.size x_t 0 in
    
    (* Generate paths *)
    let dw = Tensor.randn [batch_size; config.num_steps] ~device:config.device
             |> Tensor.mul_scalar (Float.sqrt config.dt) in
    
    let rec step_forward s acc =
      if s >= config.num_steps then List.rev acc
      else
        let curr_t = Tensor.full [batch_size] 
          (float_of_int s *. config.dt) ~device:config.device in
        let curr_x = Tensor.add x_t (Tensor.select dw ~dim:1 ~index:s) in
        
        let input = Tensor.cat [curr_t; curr_x] ~dim:1 in
        let u = Network.forward network input in
        let grad_u = Network.gradient network input in
        
        step_forward (s + 1) ((curr_x, u, grad_u) :: acc)
    in
    step_forward 0 []

  let compute_next_iterate ~pde ~network ~paths ~config =
    let x_T, _, _ = List.hd paths in
    let terminal_values = pde.terminal_condition x_T in
    
    let integral_term = List.fold_left (fun acc (x_s, u_s, grad_u_s) ->
      let f_s = pde.nonlinearity u_s grad_u_s in
      Tensor.add acc (Tensor.mul_scalar f_s config.dt)
    ) (Tensor.zeros_like terminal_values) paths in
    
    Tensor.add terminal_values integral_term
end

module Training = struct
  type config = {
    batch_size : int;
    learning_rate : float;
    num_epochs : int;
    num_mc_samples : int;
    picard_iterations : int;
    dt : float;
    lambda : float;
  }

  let train ~pde ~network ~config ~device =
    let optimizer = Optimizer.adam (Var_store.vars network.Network.layers) 
                     ~learning_rate:config.learning_rate in
    
    for k = 0 to config.picard_iterations - 1 do
      Printf.printf "Picard iteration %d\n" k;
      
      for epoch = 0 to config.num_epochs - 1 do
        (* Generate batch *)
        let x_batch = Tensor.randn [config.batch_size; pde.dim] ~device in
        let t_batch = Tensor.rand [config.batch_size; 1] ~device 
                     |> Tensor.mul_scalar time_horizon in
        
        (* Compute values and gradients *)
        let true_values = MC.feynman_kac_estimate 
          ~pde ~network ~x_t:x_batch ~t:t_batch 
          ~dt:config.dt ~num_samples:config.num_mc_samples ~device in
        
        let true_grads = MC.gradient_finite_var
          ~pde ~network ~x_t:x_batch ~t:t_batch
          ~dt:config.dt ~num_samples:config.num_mc_samples in
        
        (* Forward pass *)
        let input = Tensor.cat [t_batch; x_batch] ~dim:1 in
        let pred_values = Network.forward network input in
        let pred_grads = Network.gradient network input in
        
        (* Compute loss *)
        let value_loss = Tensor.mse_loss pred_values true_values in
        let grad_loss = Tensor.mse_loss pred_grads true_grads in
        let loss = Tensor.add value_loss (Tensor.mul_scalar grad_loss config.lambda) in
        
        (* Backward and optimize *)
        Optimizer.zero_grad optimizer;
        Tensor.backward loss;
        Optimizer.step optimizer;
        
        if epoch mod 10 = 0 then
          Printf.printf "Epoch %d Loss: %f\n" epoch (Tensor.float_value loss)
      done
    done
end

module Optimization = struct
  module Parallel = struct
    type batch_config = {
      total_size : int;
      device_batch_size : int;
      num_gpus : int;
      main_device : Device.t;
    }

    let distribute_batch ~config ~data =
      let batches = ref [] in
      let device_idx = ref 0 in
      
      for i = 0 to config.total_size / config.device_batch_size - 1 do
        let start_idx = i * config.device_batch_size in
        let device = Device.Cuda (!device_idx) in
        let batch = Tensor.narrow data ~dim:0 ~start:start_idx 
                                 ~length:config.device_batch_size
                   |> Tensor.to_device device in
        batches := (batch, device) :: !batches;
        device_idx := (!device_idx + 1) mod config.num_gpus
      done;
      !batches

    let parallel_forward ~network ~batches =
      List.map (fun (batch, device) ->
        let network_device = { network with device } in
        Network.forward network_device batch
      ) batches
  end

  module Memory = struct
    let checkpoint_forward ~network ~input ~checkpoint_layers =
      let rec forward layers x checkpoints =
        match layers with
        | [] -> x, checkpoints
        | layer :: rest ->
            let y = nn_forward layer x in
            let checkpoints = 
              if List.mem layer checkpoint_layers then
                (layer, x) :: checkpoints
              else checkpoints
            in
            forward rest y checkpoints
      in
      forward network.Network.layers input []

    let stream_batches ~total_size ~batch_size ~f =
      let rec process_batch start acc =
        if start >= total_size then 
          List.rev acc
        else
          let curr_batch_size = min batch_size (total_size - start) in
          let batch = f start curr_batch_size in
          process_batch (start + curr_batch_size) (batch :: acc)
      in
      process_batch 0 []
  end
end