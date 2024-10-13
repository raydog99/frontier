open Torch

type dual_params = {
  model_params: Model.model_params;
  utility: Utility.Utility;
  network: Neural_network.Network;
}

let create_dual_params ~model_params ~utility ~network_architecture =
  let network = network_architecture 3 50 1 in
  {model_params; utility; network}

let dual_loss params t y r =
  let module U = (val params.utility : Utility.Utility) in
  let module N = (val params.network : Neural_network.Network) in

  let v = N.model (Tensor.cat [t; y; r] ~dim:1) in
  let v_t = Tensor.grad v [t] in
  let v_y = Tensor.grad v [y] in
  let v_yy = Tensor.grad v_y [y] in
  let v_r = Tensor.grad v [r] in
  let v_rr = Tensor.grad v_r [r] in
  let v_yr = Tensor.grad v_y [r] in

  let mp = params.model_params in

  let pde_loss =
    Tensor.(v_t - (y * mp.alpha * v_y) + 
            (float_vec [|0.5|] * (y ** float_vec [|2.|]) * (mp.theta ** float_vec [|2.|]) * v_yy) +
            (mp.a * r * v_r) + 
            (float_vec [|0.5|] * (mp.b ** float_vec [|2.|]) * (r ** float_vec [|2.|]) * v_rr) -
            (mp.rho * mp.theta * mp.b * y * r * v_yr) -
            (float_vec [|0.5|] * ((float_vec [|1.|] - mp.rho ** float_vec [|2.|]) * 
                                  (mp.b * r * v_yr) ** float_vec [|2.|]) / v_yy))
  in

  let terminal_loss = Tensor.(U.evaluate y - v) in

  Tensor.(pde_loss ** float_vec [|2.|] + terminal_loss ** float_vec [|2.|])

let train_dual params optimizer data_loader num_epochs =
  let module N = (val params.network : Neural_network.Network) in

  for epoch = 1 to num_epochs do
    let batches = data_loader () in
    List.iter (fun (t_batch, y_batch, r_batch, _) ->
      Optimizer.zero_grad optimizer;
      let loss = dual_loss params t_batch y_batch r_batch in
      let loss_value = Tensor.mean loss in
      Tensor.backward loss_value;
      Optimization.gradient_clipping optimizer 1.0;
      Optimizer.step optimizer;
    ) batches;

    if epoch mod 10 = 0 then
      Printf.printf "Epoch %d complete\n" epoch
  done

let compute_primal_from_dual params t x r =
  let module N = (val params.network : Neural_network.Network) in
  let input = Tensor.cat [t; x; r] ~dim:1 in
  let v = N.model input in
  let v_y = Tensor.grad v [x] in
  Tensor.(v - (x * v_y))