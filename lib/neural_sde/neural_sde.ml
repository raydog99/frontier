open Torch
open Types
open Utils

type t = {
  params: sde_params;
  rho: float;
  r: float;
  s0: float;
  b_v_nn: Nn.t;
  sigma_nn: Nn.t;
  sigma_v_nn: Nn.t;
  prior_std: float;
  zeta_nn: Nn.t;
}

let create ~hidden_dim ~rho ~r ~s0 ~prior_std =
  let device = Torch.Device.cuda_if_available () in
  let b_v_nn = create_feed_forward ~input_dim:1 ~hidden_dim ~output_dim:1 in
  let sigma_nn = create_feed_forward ~input_dim:2 ~hidden_dim ~output_dim:1 in
  let sigma_v_nn = create_feed_forward ~input_dim:1 ~hidden_dim ~output_dim:1 in
  let zeta_nn = create_feed_forward ~input_dim:2 ~hidden_dim ~output_dim:2 in
  
  let b_v v = Nn.forward b_v_nn v in
  let sigma s v = Nn.forward sigma_nn (Tensor.cat [s; v] ~dim:1) in
  let sigma_v v = Nn.forward sigma_v_nn v in
  
  let params = { b_v; sigma; sigma_v } in
  { params; rho; r; s0; b_v_nn; sigma_nn; sigma_v_nn; prior_std; zeta_nn }

let simulate t ~num_paths ~num_steps ~dt ~measure =
  let device = Torch.Device.cuda_if_available () in
  let s0 = Tensor.full [num_paths; 1] t.s0 ~device in
  let v0 = Tensor.full [num_paths; 1] 0.04 ~device in
  
  let rec simulate_step s v step =
    if step = num_steps then (s, v)
    else
      let dw1 = Tensor.(normal [num_paths; 1] ~mean:0.0 ~std:(sqrt dt)) in
      let dw2 = Tensor.(normal [num_paths; 1] ~mean:0.0 ~std:(sqrt dt)) in
      let dw2 = Tensor.((dw1 * t.rho) + (dw2 * (sqrt (1. -. (t.rho *. t.rho))))) in
      
      let zeta = match measure with
        | `Risk_neutral -> Tensor.zeros [num_paths; 2] ~device
        | `Objective -> Nn.forward t.zeta_nn (Tensor.cat [s; v] ~dim:1)
      in
      
      let ds = Tensor.(
        (s * (t.r - (get zeta [None; 0]) * (t.params.sigma s v)) * dt) + 
        (s * (t.params.sigma s v) * dw1)
      ) in
      let dv = Tensor.(
        ((t.params.b_v v + (get zeta [None; 1]) * (t.params.sigma_v v)) * dt) +
        (t.params.sigma_v v * dw2)
      ) in
      
      let s_next = Tensor.(s + ds) in
      let v_next = Tensor.(v + dv) in
      
      simulate_step s_next v_next (step + 1)
  in
  
  simulate_step s0 v0 0

let price_option_with_control_variate t ~option_data ~num_paths =
  let num_steps = int_of_float (option_data.maturity /. 0.01) in
  let dt = option_data.maturity /. float_of_int num_steps in
  
  let s_t, _ = simulate t ~num_paths ~num_steps ~dt ~measure:`Risk_neutral in
  let payoff = Tensor.(max (s_t - (float option_data.strike)) (float 0.0)) in
  
  let h_nn = create_feed_forward ~input_dim:1 ~hidden_dim:50 ~output_dim:1 in
  let h s = Nn.forward h_nn s in
  
  let control_variate = Tensor.(
    sum (h s_t - (h (Tensor.full [num_paths; 1] t.s0 ~device:(Tensor.device s_t))))
  ) in
  
  let discounted_payoff = Tensor.(payoff * (exp (float (-t.r *. option_data.maturity)))) in
  let price = Tensor.((mean discounted_payoff) - (mean control_variate)) |> Tensor.to_float0_exn in
  
  price

let loss t ~option_data ~num_paths =
  let prices = List.map (fun od -> price_option_with_control_variate t ~option_data:od ~num_paths) option_data in
  let market_prices = List.map (fun od -> od.market_price) option_data in
  let weights = List.map (fun od -> od.weight) option_data in
  
  let squared_errors = List.map2 (fun p mp -> (p -. mp) ** 2.0) prices market_prices in
  let weighted_errors = List.map2 ( *. ) squared_errors weights in
  List.fold_left ( +. ) 0.0 weighted_errors

let time_series_log_likelihood t ~time_series ~dt =
  let num_steps = List.length time_series - 1 in
  let y = Tensor.of_float1 time_series in
  let v = Tensor.zeros [num_steps + 1; 1] in  (* Constant volatility *)
  
  let log_likelihood = ref 0.0 in
  for i = 0 to num_steps - 1 do
    let y_t = Tensor.slice y ~dim:0 ~start:i ~end_:(i+1) in
    let y_next = Tensor.slice y ~dim:0 ~start:(i+1) ~end_:(i+2) in
    let v_t = Tensor.slice v ~dim:0 ~start:i ~end_:(i+1) in
    
    let mu = Tensor.(
      (t.r - 0.5 * (t.params.sigma y_t v_t ** 2.)) +
      ((t.params.sigma y_t v_t) * (get (Nn.forward t.zeta_nn (Tensor.cat [y_t; v_t] ~dim:1)) [None; 0]))
    ) in
    let sigma = t.params.sigma y_t v_t in
    
    let diff = Tensor.((y_next - y_t - mu * dt) / (sigma * (sqrt (float dt)))) in
    log_likelihood := !log_likelihood +. (Tensor.to_float0_exn Tensor.(
      -0.5 * (log (float 2. *. Float.pi) + (2. * log sigma) + (diff * diff))
    ))
  done;
  
  !log_likelihood

let prior_log_prob t =
  let prior_var = t.prior_std *. t.prior_std in
  let log_prob_tensor tensor =
    let n = Tensor.numel tensor in
    let log_prob = Tensor.(
      -0.5 * (n * log (float 2. *. Float.pi *. prior_var) + (sum (tensor * tensor) / float prior_var))
    ) in
    Tensor.to_float0_exn log_prob
  in
  let b_v_log_prob = log_prob_tensor (Nn.flat_parameters t.b_v_nn) in
  let sigma_log_prob = log_prob_tensor (Nn.flat_parameters t.sigma_nn) in
  let sigma_v_log_prob = log_prob_tensor (Nn.flat_parameters t.sigma_v_nn) in
  b_v_log_prob +. sigma_log_prob +. sigma_v_log_prob

let langevin_step t ~step_size ~sigma =
  let update_params nn =
    List.iter (fun p ->
      let grad = Tensor.grad p in
      let noise = Tensor.randn (Tensor.shape p) ~device:(Tensor.device p) in
      Tensor.(p -= (grad * float step_size + (noise * float (sqrt (2. *. step_size *. sigma)))))
    ) (Nn.parameters nn)
  in
  
  update_params t.b_v_nn;
  update_params t.sigma_nn;
  update_params t.sigma_v_nn;
  update_params t.zeta_nn;
  
  Tensor.zero_grad (Nn.flat_parameters t.b_v_nn);
  Tensor.zero_grad (Nn.flat_parameters t.sigma_nn);
  Tensor.zero_grad (Nn.flat_parameters t.sigma_v_nn);
  Tensor.zero_grad (Nn.flat_parameters t.zeta_nn)

let bayesian_calibrate t ~option_data ~time_series ~num_epochs ~step_size ~sigma ~num_paths =
  let rec train epoch t_current =
    if epoch = num_epochs then t_current
    else
      begin
        let t_next = { t_current with
          b_v_nn = Nn.copy t_current.b_v_nn;
          sigma_nn = Nn.copy t_current.sigma_nn;
          sigma_v_nn = Nn.copy t_current.sigma_v_nn;
          zeta_nn = Nn.copy t_current.zeta_nn;
        } in
        
        let option_loss = loss t_next ~option_data ~num_paths in
        let time_series_ll = time_series_log_likelihood t_next ~time_series ~dt:0.01 in
        let prior_log_prob_value = prior_log_prob t_next in
        
        let total_loss = Tensor.(
          add (float option_loss)
            (add (neg (float time_series_ll))
                 (neg (float prior_log_prob_value)))
        ) in
        
        Tensor.backward total_loss;
        
        langevin_step t_next ~step_size ~sigma;
        
        if epoch mod 100 = 0 then
          Printf.printf "Epoch %d: Option Loss = %f, Time Series LL = %f\n"
            epoch option_loss time_series_ll;
        
        train (epoch + 1) t_next
      end
  in
  
  train 0 t