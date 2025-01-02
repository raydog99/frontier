open Torch

type pinn_params = {
  model_params: Model.model_params;
  utility: Utility.Utility;
  network: Neural_network.Network;
}

let create_pinn_params ~model_params ~utility ~network_architecture =
  let network = network_architecture 3 50 1 in
  {model_params; utility; network}

let pinn_loss params t x r =
  let module U = (val params.utility : Utility.Utility) in
  let module N = (val params.network : Neural_network.Network) in
  
  let v = N.model (Tensor.cat [t; x; r] ~dim:1) in
  let v_t = Tensor.grad v [t] in
  let v_x = Tensor.grad v [x] in
  let v_xx = Tensor.grad v_x [x] in
  let v_r = Tensor.grad v [r] in
  let v_rr = Tensor.grad v_r [r] in
  let v_xr = Tensor.grad v_x [r] in

  let mp = params.model_params in
  
  let pde_loss =
    Tensor.(v_t + (x * mp.alpha * v_x) + (mp.a * r * v_r) +
            (float_vec [|0.5|] * (mp.b ** float_vec [|2.|]) * (r ** float_vec [|2.|]) * v_rr) -
            (float_vec [|0.5|] * ((mp.theta * v_x + mp.rho * mp.b * r * v_xr) ** float_vec [|2.|]) / v_xx))
  in

  let terminal_loss = Tensor.(U.evaluate (x - r) - v) in

  Tensor.(pde_loss ** float_vec [|2.|] + terminal_loss ** float_vec [|2.|])

let train_pinn params optimizer data_loader num_epochs =
  let module N = (val params.network : Neural_network.Network) in

  for epoch = 1 to num_epochs do
    let batches = data_loader () in
    List.iter (fun (t_batch, x_batch, r_batch, _) ->
      Optimizer.zero_grad optimizer;
      let loss = pinn_loss params t_batch x_batch r_batch in
      let loss_value = Tensor.mean loss in
      Tensor.backward loss_value;
      Optimization.gradient_clipping optimizer 1.0;
      Optimizer.step optimizer;
    ) batches;

    if epoch mod 10 = 0 then
      Printf.printf "Epoch %d complete\n" epoch
  done

let evaluate_pinn params t x r =
  let module N = (val params.network : Neural_network.Network) in
  N.model (Tensor.cat [t; x; r] ~dim:1)

let train_pinn_with_concavification params optimizer data_loader num_epochs =
  let module N = (val params.network : Neural_network.Network) in
  let module U = (val params.utility : Utility.Utility) in

  for epoch = 1 to num_epochs do
    let batches = data_loader () in
    List.iter (fun (t_batch, x_batch, r_batch, _) ->
      Optimizer.zero_grad optimizer;
      let v_pred = N.model (Tensor.cat [t_batch; x_batch; r_batch] ~dim:1) in
      let v_concave = Concavification.concavify params.utility x_batch r_batch in
      let loss = Tensor.mse_loss v_pred v_concave in
      Tensor.backward loss;
      Optimization.gradient_clipping optimizer 1.0;
      Optimizer.step optimizer;
    ) batches;

    if epoch mod 10 = 0 then
      Printf.printf "Epoch %d complete\n" epoch
  done