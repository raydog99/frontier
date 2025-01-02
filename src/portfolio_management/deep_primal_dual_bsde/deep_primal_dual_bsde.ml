open Torch
open Stdio
open Lwt
open Lwt_list

module Logger = struct
  let log_file = ref (Out_channel.create "deep_primal_dual_bsde.log")
  
  let log message =
    let timestamp = Unix.gettimeofday () |> Unix.localtime in
    Printf.fprintf !log_file "[%04d-%02d-%02d %02d:%02d:%02d] %s\n"
      (timestamp.tm_year + 1900) (timestamp.tm_mon + 1) timestamp.tm_mday
      timestamp.tm_hour timestamp.tm_min timestamp.tm_sec message;
    Out_channel.flush !log_file

  let close () =
    Out_channel.close !log_file
end

module Option_Type = struct
  type t =
    | GeometricBasketCall
    | MaxCall
    | StrangleSpreadBasket

  let payoff option_type params x =
    match option_type with
    | GeometricBasketCall ->
        let open Tensor in
        let geo_mean = mean x ~dim:[1] |> exp in
        relu (sub geo_mean (Scalar.f params.strike))
    | MaxCall ->
        let open Tensor in
        let max_price = max x ~dim:[1] ~keepdim:false in
        relu (sub max_price (Scalar.f params.strike))
    | StrangleSpreadBasket ->
        let open Tensor in
        let geo_mean = mean x ~dim:[1] |> exp in
        let call_payoff = relu (sub geo_mean (Scalar.f params.strike)) in
        let put_payoff = relu (sub (Scalar.f params.strike) geo_mean) in
        add call_payoff put_payoff
end

module Option_Pricing = struct
  type params = {
    strike: float;
    risk_free_rate: float;
    volatility: float;
    dividend_rate: float;
    option_type: Option_Type.t;
  }

  let simulate_paths_parallel params problem_params num_paths num_workers =
    let dt = problem_params.maturity /. float_of_int problem_params.num_time_steps in
    let sqrt_dt = sqrt dt in
    let drift = Scalar.f (params.risk_free_rate -. params.dividend_rate -. 0.5 *. params.volatility ** 2.) in
    let vol = Scalar.f (params.volatility *. sqrt_dt) in
    
    let simulate_chunk _ =
      let chunk_size = num_paths / num_workers in
      let init_price = Tensor.ones [chunk_size; problem_params.dim] in
      let brownian_increments = Tensor.randn [chunk_size; problem_params.num_time_steps; problem_params.dim] in
      
      let simulate_step prev_price increment =
        let open Tensor in
        prev_price * exp (drift * Scalar.f dt + vol * increment)
      in
      
      Tensor.scan ~dim:1 ~f:simulate_step brownian_increments ~init:init_price
    in
    
    Lwt_list.map_p (fun _ -> Lwt.return (simulate_chunk ())) (List.init num_workers (fun _ -> ()))
    >|= Tensor.cat ~dim:0

  let black_scholes_price params problem_params =
    let t = problem_params.maturity in
    let r = params.risk_free_rate in
    let q = params.dividend_rate in
    let sigma = params.volatility in
    let k = params.strike in
    let s0 = 100.0 in
    
    let d1 = (log (s0 /. k) +. (r -. q +. 0.5 *. sigma ** 2.) *. t) /. (sigma *. sqrt t) in
    let d2 = d1 -. sigma *. sqrt t in
    
    let nd1 = 0.5 *. (1. +. Tensor.erf (Tensor.of_float [|d1 /. sqrt 2.|])) |> Tensor.to_float0_exn in
    let nd2 = 0.5 *. (1. +. Tensor.erf (Tensor.of_float [|d2 /. sqrt 2.|])) |> Tensor.to_float0_exn in
    
    s0 *. exp (-.q *. t) *. nd1 -. k *. exp (-.r *. t) *. nd2
end

module Optimal_Stopping_Problem = struct
  type problem_params = {
    dim: int;
    maturity: float;
    num_time_steps: int;
  }

  type t = {
    params: problem_params;
    option_params: Option_Pricing.params;
    value_networks: (int * Layer.t list) list;
    gradient_networks: (int * Layer.t list) list;
    hedging_networks: (int * Layer.t list) list;
  }

  let create params option_params =
    let nn_params = { layers = 2; neurons = 64 } in
    let create_networks () =
      List.init params.num_time_steps (fun k ->
        (k, FeedForward_NN.create nn_params params.dim 1),
        (k, FeedForward_NN.create nn_params params.dim params.dim),
        (k, FeedForward_NN.create nn_params params.dim params.dim)
      )
    in
    let value_nets, grad_nets, hedge_nets = List.split3 (create_networks ()) in
    { params; option_params; value_networks = value_nets; gradient_networks = grad_nets; hedging_networks = hedge_nets }

  let forward_network networks x k =
    let network = List.assoc k networks in
    Layer.forward network x

  let mixed_type_values problem x_k k =
    let open Tensor in
    let immediate_payoff = Option_Type.payoff problem.option_params.option_type problem.option_params x_k in
    let continuation_value = forward_network problem.value_networks x_k k in
    Tensor.(add (mul immediate_payoff (Scalar.f 0.5)) (mul continuation_value (Scalar.f 0.5)))

  let bsde_loss problem x_k tau_k_plus_1 x_tau_k_plus_1 delta_w_k k delta_m_k_plus_1 =
    let open Tensor in
    let c_k = mixed_type_values problem x_k k in
    let g_k = forward_network problem.gradient_networks x_k k in
    let h_k = forward_network problem.hedging_networks x_k k in
    let g_tau = Option_Type.payoff problem.option_params.option_type problem.option_params x_tau_k_plus_1 in
    let delta_m_k = Tensor.matmul g_k (Tensor.transpose delta_w_k ~dim0:(-1) ~dim1:(-2)) in
    let hedging_loss = Tensor.(mean (pow (sub g_k h_k) (Scalar.f 2.))) in
    Tensor.(add (mean (pow (c_k - g_tau + delta_m_k + delta_m_k_plus_1) (Scalar.f 2.))) (mul hedging_loss (Scalar.f 0.1)))

  let train problem num_epochs num_steps batch_size learning_rate =
    let all_params = 
      List.concat (List.map (fun (_, net) -> Layer.parameters net) problem.value_networks) @
      List.concat (List.map (fun (_, net) -> Layer.parameters net) problem.gradient_networks) @
      List.concat (List.map (fun (_, net) -> Layer.parameters net) problem.hedging_networks)
    in
    let optimizer = Optimizer.adam all_params ~lr:learning_rate in
    
    Logger.log "Starting training";
    for epoch = 1 to num_epochs do
      let%lwt paths, brownian_increments = 
        Option_Pricing.simulate_paths_parallel 
          problem.option_params problem.params batch_size 4 
      in
      
      let delta_m_cumulative = Tensor.zeros [batch_size; 1] in
      
      for k = problem.params.num_time_steps - 1 downto 0 do
        for step = 1 to num_steps do
          Optimizer.zero_grad optimizer;
          
          let x_k = Tensor.narrow paths ~dim:1 ~start:k ~length:1 |> Tensor.squeeze ~dim:1 in
          let tau_k_plus_1 = Tensor.full [batch_size; 1] (float_of_int (k + 1) *. problem.params.maturity /. float_of_int problem.params.num_time_steps) in
          let x_tau_k_plus_1 = Tensor.narrow paths ~dim:1 ~start:(k+1) ~length:1 |> Tensor.squeeze ~dim:1 in
          let delta_w_k = Tensor.narrow brownian_increments ~dim:1 ~start:k ~length:1 |> Tensor.squeeze ~dim:1 in
          
          let loss = bsde_loss problem x_k tau_k_plus_1 x_tau_k_plus_1 delta_w_k k delta_m_cumulative in
          
          Tensor.backward loss;
          Optimizer.step optimizer;
          
          let g_k = forward_network problem.gradient_networks x_k k in
          let delta_m_k = Tensor.matmul g_k (Tensor.transpose delta_w_k ~dim0:(-1) ~dim1:(-2)) in
          Tensor.add_ delta_m_cumulative delta_m_k;
          
          if step mod 100 = 0 then
            Logger.log (Printf.sprintf "Epoch %d, Time step %d, Step %d, Loss: %f" epoch k step (Tensor.to_float0_exn loss))
        done
      done;
      Lwt.return_unit
    done;
    Logger.log "Training completed";
    Lwt.return_unit

  let compute_lower_bound problem paths =
    let num_paths = (Tensor.shape paths).(0) in
    let payoffs = Tensor.zeros [num_paths] in
    let stopping_times = Tensor.full [num_paths] (float_of_int problem.params.num_time_steps) in
    
    for k = problem.params.num_time_steps - 1 downto 0 do
      let x_k = Tensor.narrow paths ~dim:1 ~start:k ~length:1 |> Tensor.squeeze ~dim:1 in
      let immediate_payoff = Option_Type.payoff problem.option_params.option_type problem.option_params x_k in
      let continuation_value = forward_network problem.value_networks x_k k in
      
      let should_stop = Tensor.(gt immediate_payoff continuation_value) in
      let new_payoffs = Tensor.where should_stop immediate_payoff payoffs in
      let new_stopping_times = Tensor.where should_stop (Tensor.full_like stopping_times (float_of_int k)) stopping_times in
      
      Tensor.copy_ payoffs new_payoffs;
      Tensor.copy_ stopping_times new_stopping_times
    done;
    
    let mean_payoff = Tensor.mean payoffs in
    let std_payoff = Tensor.std payoffs ~dim:[0] ~unbiased:true ~keepdim:false in
    let confidence_interval = Tensor.(mul (div std_payoff (sqrt (Scalar.f (float_of_int num_paths)))) (Scalar.f 1.96)) in
    
    (Tensor.to_float0_exn mean_payoff, Tensor.to_float0_exn confidence_interval)

  let compute_upper_bound problem paths brownian_increments finer_grid_factor =
    let num_paths = (Tensor.shape paths).(0) in
    let dt = problem.params.maturity /. float_of_int problem.params.num_time_steps in
    let finer_dt = dt /. float_of_int finer_grid_factor in
    
    let martingale_increments = Tensor.zeros [num_paths; problem.params.num_time_steps] in
    let payoffs = Tensor.zeros [num_paths; problem.params.num_time_steps + 1] in
    
    for k = 0 to problem.params.num_time_steps do
      let x_k = Tensor.narrow paths ~dim:1 ~start:k ~length:1 |> Tensor.squeeze ~dim:1 in
      let payoff_k = Option_Type.payoff problem.option_params.option_type problem.option_params x_k in
      Tensor.copy_ (Tensor.narrow payoffs ~dim:1 ~start:k ~length:1) payoff_k
    done;
    
    for k = 0 to problem.params.num_time_steps - 1 do
      let x_k = Tensor.narrow paths ~dim:1 ~start:k ~length:1 |> Tensor.squeeze ~dim:1 in
      let grad_k = forward_network problem.gradient_networks x_k k in
      
      let increment = Tensor.zeros [num_paths] in
      for j = 0 to finer_grid_factor - 1 do
        let t = float_of_int k *. dt +. float_of_int j *. finer_dt in
        let x_t = Tensor.narrow paths ~dim:1 ~start:(k * finer_grid_factor + j) ~length:1 |> Tensor.squeeze ~dim:1 in
        let dw = Tensor.randn [num_paths; problem.params.dim] |> Tensor.mul_scalar (Scalar.f (sqrt finer_dt)) in
        let delta_m = Tensor.sum (Tensor.mul grad_k dw) ~dim:[1] in
        Tensor.add_ increment delta_m
      done;
      
      Tensor.copy_ (Tensor.narrow martingale_increments ~dim:1 ~start:k ~length:1) increment
    done;
    
    let cumulative_martingale = Tensor.cumsum martingale_increments ~dim:1 in
    
    let max_diff = Tensor.max (Tensor.sub payoffs cumulative_martingale) ~dim:[1] |> fst in
    let upper_bound = Tensor.mean max_diff in
    let std_upper_bound = Tensor.std max_diff ~dim:[0] ~unbiased:true ~keepdim:false in
    let confidence_interval = Tensor.(mul (div std_upper_bound (sqrt (Scalar.f (float_of_int num_paths)))) (Scalar.f 1.96)) in
    
    (Tensor.to_float0_exn upper_bound, Tensor.to_float0_exn confidence_interval)

  let compute_hedging_ratios problem paths =
    let num_paths = (Tensor.shape paths).(0) in
    let hedging_ratios = Tensor.zeros [num_paths; problem.params.num_time_steps; problem.params.dim] in
    
    for k = 0 to problem.params.num_time_steps - 1 do
      let x_k = Tensor.narrow paths ~dim:1 ~start:k ~length:1 |> Tensor.squeeze ~dim:1 in
      let h_k = forward_network problem.hedging_networks x_k k in
      Tensor.copy_ (Tensor.narrow hedging_ratios ~dim:1 ~start:k ~length:1) h_k
    done;
    
    hedging_ratios

  let analyze_model problem num_paths =
    let%lwt paths, brownian_increments = 
      Option_Pricing.simulate_paths_parallel 
        problem.option_params problem.params num_paths 4 
    in
    let lower_bound, lower_ci = compute_lower_bound problem paths in
    let upper_bound, upper_ci = compute_upper_bound problem paths brownian_increments 10 in
    let hedging_ratios = compute_hedging_ratios problem paths in
    
    let bs_price = Option_Pricing.black_scholes_price problem.option_params problem.params in
    
    let relative_error_lower = abs_float (lower_bound -. bs_price) /. bs_price in
    let relative_error_upper = abs_float (upper_bound -. bs_price) /. bs_price in
    
    Logger.log (Printf.sprintf "Model Analysis:");
    Logger.log (Printf.sprintf "Black-Scholes Price: %f" bs_price);
    Logger.log (Printf.sprintf "Lower Bound: %f ± %f (Relative Error: %.2f%%)" lower_bound lower_ci (relative_error_lower *. 100.));
    Logger.log (Printf.sprintf "Upper Bound: %f ± %f (Relative Error: %.2f%%)" upper_bound upper_ci (relative_error_upper *. 100.));
    
    let avg_hedging_ratio = Tensor.(mean hedging_ratios ~dim:[0; 1]) in
    Logger.log (Printf.sprintf "Average Hedging Ratio: %s" (Tensor.to_string avg_hedging_ratio));
    
    let exercise_boundary = Tensor.zeros [problem.params.num_time_steps] in
    for k = 0 to problem.params.num_time_steps - 1 do
      let t = float_of_int k *. problem.params.maturity /. float_of_int problem.params.num_time_steps in
      let x = Tensor.linspace ~start:0. ~end_:200. ~steps:1000 in
      let values = forward_network problem.value_networks x k in
      let payoffs = Option_Type.payoff problem.option_params.option_type problem.option_params x in
      let diff = Tensor.(sub values payoffs) in
      let idx = Tensor.argmin diff ~dim:0 ~keepdim:false in
      Tensor.copy_ (Tensor.narrow exercise_boundary ~dim:0 ~start:k ~length:1) (Tensor.narrow x ~dim:0 ~start:(Tensor.to_int0_exn idx) ~length:1)
    done;
    
    Lwt.return (lower_bound, upper_bound, hedging_ratios, exercise_boundary, relative_error_lower, relative_error_upper)
end

module Experiment = struct
  type experiment_params = {
    dim: int;
    maturity: float;
    num_time_steps: int;
    strike: float;
    risk_free_rate: float;
    volatility: float;
    dividend_rate: float;
    option_type: Option_Type.t;
    num_epochs: int;
    num_steps: int;
    batch_size: int;
    learning_rate: float;
    num_paths: int;
  }

  let run_experiment params =
    let problem_params = { 
      dim = params.dim; 
      maturity = params.maturity; 
      num_time_steps = params.num_time_steps 
    } in
    let option_params = { 
      strike = params.strike;
      risk_free_rate = params.risk_free_rate;
      volatility = params.volatility;
      dividend_rate = params.dividend_rate;
      option_type = params.option_type;
    } in
    let problem = Optimal_Stopping_Problem.create problem_params option_params in
    let%lwt () = Optimal_Stopping_Problem.train problem params.num_epochs params.num_steps params.batch_size params.learning_rate in
    Optimal_Stopping_Problem.analyze_model problem params.num_paths

  let compare_experiments experiments =
    let%lwt results = Lwt_list.map_p (fun params -> run_experiment params) experiments in
    
    let plot_prices results =
      let x = List.mapi (fun i _ -> float_of_int i) results in
      let lower_bounds, upper_bounds, _, _, _, _ = List.split6 results in
      ()
    in
    
    let plot_exercise_boundary result_idx =
      let _, _, _, exercise_boundary, _, _ = List.nth results result_idx in
      let x = Tensor.linspace ~start:0. ~end_:(List.nth experiments result_idx).maturity ~steps:(List.nth experiments result_idx).num_time_steps in
      ()
    in
    
    let plot_error_analysis results =
      let _, _, _, _, rel_errors_lower, rel_errors_upper = List.split6 results in
      ()
    in
    
    plot_prices results;
    plot_exercise_boundary 0;
    plot_error_analysis results;
    
    Lwt.return results
end

module Tests = struct
  let test_option_payoff () =
    let params = { Option_Pricing.strike = 100.; risk_free_rate = 0.05; volatility = 0.2; dividend_rate = 0.; option_type = Option_Type.GeometricBasketCall } in
    let x = Tensor.of_float2 [[95.; 105.]; [100.; 100.]; [110.; 90.]] in
    let payoff = Option_Type.payoff params.option_type params x in
    let expected = Tensor.of_float1 [0.; 0.; 0.] in
    assert (Tensor.allclose payoff expected ~rtol:1e-5 ~atol:1e-8);
    Printf.printf "Option payoff test passed.\n"

  let run_all_tests () =
    test_option_payoff ();
    Printf.printf "All tests passed.\n"
end

let main () =
  Lwt_main.run begin
    try%lwt
      Tests.run_all_tests ();

      let base_params = {
        Experiment.
        dim = 20;
        maturity = 2.0;
        num_time_steps = 50;
        strike = 100.0;
        risk_free_rate = 0.05;
        volatility = 0.2;
        dividend_rate = 0.0;
        option_type = Option_Type.GeometricBasketCall;
        num_epochs = 100;
        num_steps = 1000;
        batch_size = 256;
        learning_rate = 1e-3;
        num_paths = 100000;
      } in
      
      let experiments = [
        base_params;
        { base_params with volatility = 0.1 };
        { base_params with volatility = 0.3 };
        { base_params with risk_free_rate = 0.03 };
        { base_params with risk_free_rate = 0.07 };
        { base_params with dim = 30 };
        { base_params with dim = 50 };
        { base_params with option_type = Option_Type.MaxCall };
        { base_params with option_type = Option_Type.StrangleSpreadBasket };
      ] in
      
      let%lwt results = Experiment.compare_experiments experiments in
      
      List.iteri (fun i (lower_bound, upper_bound, _, _, rel_error_lower, rel_error_upper) ->
        Printf.printf "Experiment %d results:\n" i;
        Printf.printf "  Lower bound: %f (Relative Error: %.2f%%)\n" lower_bound (rel_error_lower *. 100.);
        Printf.printf "  Upper bound: %f (Relative Error: %.2f%%)\n" upper_bound (rel_error_upper *. 100.);
      ) results;
      
      Lwt.return_unit
    with
    | e ->
      Logger.log (Printf.sprintf "Error in main function: %s" (Printexc.to_string e));
      prerr_endline "An error occurred. Check the log file for details.";
      Lwt.return_unit
  end