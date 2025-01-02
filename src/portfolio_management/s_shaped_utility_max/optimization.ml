open Torch

let adam_with_lr_scheduler optimizer initial_lr num_iterations =
  let decay_rate = 0.95 in
  let decay_steps = 1000 in

  for i = 1 to num_iterations do
    let current_lr = initial_lr *. (decay_rate ** (float_of_int i /. float_of_int decay_steps)) in
    Optimizer.set_learning_rate optimizer current_lr;
    Optimizer.step optimizer;

    if i mod 1000 = 0 then
      Printf.printf "Iteration %d, Learning Rate: %f\n" i current_lr
  done

let gradient_clipping optimizer max_norm =
  let params = Optimizer.get_parameters optimizer in
  let grads = List.map Tensor.grad params in
  let total_norm = 
    List.fold_left (fun acc grad -> 
      acc +. Tensor.float_value (Tensor.sum (Tensor.square grad))
    ) 0. grads
    |> sqrt
  in
  let clip_coef = min (max_norm /. total_norm) 1. in
  List.iter2 (fun param grad ->
    Tensor.mul_inplace grad (Tensor.float_vec [|clip_coef|]);
    Tensor.set_grad param grad
  ) params grads

let find_optimal_control params t x r =
  let module N = (val params.Pinn.network : Neural_network.Network) in
  let input = Tensor.cat [t; x; r] ~dim:1 in
  let v = N.model input in
  let v_x = Tensor.grad v [x] in
  let v_xx = Tensor.grad v_x [x] in
  let v_xr = Tensor.grad v_x [r] in

  let mp = params.Pinn.model_params in
  Tensor.((neg (mp.Model.theta * v_x + mp.Model.rho * mp.Model.b * r * v_xr)) / (mp.Model.sigma * x * v_xx))

let compute_wealth_process params initial_wealth optimal_control t_steps =
  let dt = params.Pinn.model_params.Model.t /. float_of_int t_steps in
  let sqrt_dt = sqrt dt in
  
  let rec compute t wealth acc =
    if t >= params.Pinn.model_params.Model.t then List.rev acc
    else
      let dw = Tensor.randn [1] in
      let control = optimal_control t wealth in
      let drift = Tensor.((params.Pinn.model_params.Model.alpha * wealth) + (control * wealth * params.Pinn.model_params.Model.theta * params.Pinn.model_params.Model.sigma)) in
      let diffusion = Tensor.(control * wealth * params.Pinn.model_params.Model.sigma * dw) in
      let new_wealth = Tensor.(wealth + (drift * float_vec [|dt|]) + (diffusion * float_vec [|sqrt_dt|])) in
      compute (t +. dt) new_wealth (new_wealth :: acc)
  in
  
  compute 0. initial_wealth [initial_wealth]