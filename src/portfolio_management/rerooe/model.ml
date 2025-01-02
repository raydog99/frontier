open Torch
open Lwt
open Cohttp_lwt_unix
open Yojson.Basic.Util

module Simulation_config = struct
  type t = {
    model_type: [`Model1 | `Model2];
    strategy: Strategy.t;
    optimizer: Optimizer.t;
    market_impact: Market_impact.t;
    transaction_costs: Transaction_costs.t;
    params: Params.t;
    x0: float;
    s0: float;
    num_paths: int;
    dt: float;
    num_steps: int;
  }

  let create ~model_type ~strategy ~optimizer ~market_impact ~transaction_costs ~params ~x0 ~s0 ~num_paths ~dt ~num_steps =
    { model_type; strategy; optimizer; market_impact; transaction_costs; params; x0; s0; num_paths; dt; num_steps }
end

module User_interface = struct
  let create_default_config () =
    Simulation_config.create
      ~model_type:`Model1
      ~strategy:Strategy.TWAP
      ~optimizer:(Optimizer.GradientDescent { learning_rate = 0.01 })
      ~market_impact:(Market_impact.Linear { permanent = 0.1; temporary = 0.1 })
      ~transaction_costs:(Transaction_costs.Linear 0.001)
      ~params:(Params.create
        ~eta:2.5e-7 ~gamma:2.5e-6 ~sigma_s:0.1 ~sigma_x:1e-5 ~rho:0.3
        ~lambda:1. ~t:0. ~T:1. ~s:1e8 ~r_xx:1e6 ~r_xa:5e6 ~r_aa:9e7
        ~r_vv:1e6 ~r_va:5e6 ~kappa:1.)
      ~x0:1e6 ~s0:100. ~num_paths:1000 ~dt:0.01 ~num_steps:100

  let run_simulation config =
    let model = Model.create config.Simulation_config.x0 config.s0 config.market_impact config.transaction_costs in
    let vf = Value_function.create () in
    Value_function.initialize vf config.params config.model_type;
    
    Strategy.simulate config.strategy model config.params vf config.dt config.num_steps config.model_type

  let run_constrained_portfolio_optimization config assets initial_weights constraints optimization_method =
    let models = Array.map (fun (x0, s0, sector) -> 
      { Portfolio.model = Model.create x0 s0 config.Simulation_config.market_impact config.transaction_costs;
        weight = ref 0.;
        sector = sector }
    ) assets in
    let portfolio = Portfolio.create models initial_weights in
    let optimizer = {
      Constrained_optimizer.objective = optimization_method;
      constraints = constraints;
    } in
    Constrained_optimizer.optimize portfolio optimizer;
    Portfolio.get_weights portfolio

  let run_advanced_ml_strategy config historical_data model_type =
    let input_size = 10 (* Example: use last 10 days of data *)
    and output_size = 1 (* Predict next day's return *)
    and hidden_size = 64
    and num_layers = 3
    and dropout = 0.1 in
    
    let model = Ml.create_model model_type input_size output_size in
    
    let prepare_data prices =
      let n = Array.length prices in
      let x = ref []
      and y = ref [] in
      for i = 0 to n - input_size - 1 do
        x := Array.sub prices i input_size :: !x;
        y := [| (prices.(i + input_size) -. prices.(i + input_size - 1)) /. prices.(i + input_size - 1) |] :: !y
      done;
      (List.rev !x, List.rev !y)
    in
    
    let (x_data, y_data) = prepare_data historical_data.Backtester.prices in
    let data_loader = Torch_utils.Dataloader.of_list (List.combine x_data y_data) ~batch_size:32 in
    
    Ml.train model data_loader 100 0.001;
    
    let ml_strategy model params dt =
      let last_window = Array.sub historical_data.Backtester.prices
        (Array.length historical_data.Backtester.prices - input_size) input_size in
      let prediction = Ml.predict model (Tensor.of_float2 [| last_window |]) in
      let predicted_return = Tensor.to_float0_exn prediction in
      if predicted_return > 0. then 1. else -1.  (* Simple long/short strategy based on predicted return *)
    in
    
    let backtester = EventDrivenBacktester.create (Portfolio.create [||] [||]) in
    EventDrivenBacktester.run backtester (float_of_int (Array.length historical_data.Backtester.prices - 1));
    EventDrivenBacktester.get_results backtester

  let run_distributed_simulation configs num_workers =
    let tasks = List.map (fun config -> DistributedComputing.Simulation config) configs in
    let results = Lwt_main.run (DistributedComputing.distribute_tasks tasks num_workers) in
    let (simulations, _, _) = DistributedComputing.aggregate_results results in
    simulations

  let run_and_visualize config =
    let results = run_simulation config in
    WebInterface.run_server 8080 results
end

module CLI = struct
  let parse_args () =
    let model_type = ref `Model1 in
    let strategy = ref Strategy.TWAP in
    let num_paths = ref 1000 in
    let num_steps = ref 100 in
    let output_file = ref "simulation_results.csv" in
    
    let speclist = [
      ("--model", Arg.Symbol (["model1"; "model2"], 
        (fun s -> model_type := if s = "model1" then `Model1 else `Model2)), 
        "Specify the model type (model1 or model2)");
      ("--strategy", Arg.Symbol (["twap"; "vwap"; "optimal"], 
        (fun s -> strategy := match s with
          | "twap" -> Strategy.TWAP
          | "vwap" -> Util.create_vwap_strategy ()
          | "optimal" -> Strategy.Optimal
          | _ -> Strategy.TWAP)), 
        "Specify the trading strategy (twap, vwap, or optimal)");
      ("--paths", Arg.Set_int num_paths, "Number of simulation paths");
      ("--steps", Arg.Set_int num_steps, "Number of time steps");
      ("--output", Arg.Set_string output_file, "Output file for results (CSV format)");
    ] in
    let usage_msg = "Usage: " ^ Sys.argv.(0) ^ " [--model <model>] [--strategy <strategy>] [--paths <num>] [--steps <num>] [--output <file>]" in
    Arg.parse speclist (fun _ -> ()) usage_msg;
    (!model_type, !strategy, !num_paths, !num_steps, !output_file)

  let run () =
    let (model_type, strategy, num_paths, num_steps, output_file) = parse_args () in
    let config = User_interface.create_default_config () in
    let config = { config with
      Simulation_config.model_type = model_type;
      strategy = strategy;
      num_paths = num_paths;
      num_steps = num_steps;
    } in
    
    Printf.printf "Running simulation with the following configuration:\n";
    Printf.printf "Model: %s\n" (match model_type with `Model1 -> "Model1" | `Model2 -> "Model2");
    Printf.printf "Strategy: %s\n" (match strategy with
      | Strategy.TWAP -> "TWAP"
      | Strategy.VWAP _ -> "VWAP"
      | Strategy.Optimal -> "Optimal");
    Printf.printf "Number of paths: %d\n" num_paths;
    Printf.printf "Number of steps: %d\n" num_steps;
    Printf.printf "Output file: %s\n" output_file;

    let results = User_interface.run_simulation config in
    
    let strategy_name = match strategy with
      | Strategy.TWAP -> "TWAP"
      | Strategy.VWAP _ -> "VWAP"
      | Strategy.Optimal -> "Optimal"
    in
    Analysis.generate_report results strategy_name;

    let oc = open_out output_file in
    Printf.fprintf oc "Step,PnL,Risk,Entropy\n";
    List.iteri (fun i (pnl, risk, entropy) ->
      Printf.fprintf oc "%d,%.4f,%.4f,%.4f\n" i pnl risk entropy
    ) (Simulation.average_results results);
    close_out oc;

    Printf.printf "Results written to %s\n" output_file

  let run_advanced_ml_strategy () =
    let historical_data_file = ref ""
    and model_type = ref (Ml.LSTM { num_layers = 3; hidden_size = 64; dropout = 0.1 }) in
    
    let speclist = [
      ("--data", Arg.Set_string historical_data_file, "Historical data file for ML strategy");
      ("--model", Arg.Symbol (["lstm"; "transformer"; "random_forest"],
        (fun s -> model_type := match s with
          | "lstm" -> Ml.LSTM { num_layers = 3; hidden_size = 64; dropout = 0.1 }
          | "transformer" -> Ml.Transformer { num_layers = 3; num_heads = 8; d_model = 64; dropout = 0.1 }
          | "random_forest" -> Ml.RandomForest { num_trees = 100; max_depth = 10 }
          | _ -> failwith "Invalid model type")), 
        "Specify the ML model type");
    ] in
    
    Arg.parse speclist (fun _ -> ()) "Usage: advanced_ml_strategy [options]";
    
    if !historical_data_file = "" then
      invalid_arg "Historical data file must be specified";
    
    let config = User_interface.create_default_config () in
    let historical_data = Backtester.load_historical_data !historical_data_file in
    let results = User_interface.run_advanced_ml_strategy config historical_data !model_type in
    
    Printf.printf "Advanced ML Strategy Results:\n";
    Printf.printf "Total Return: %.2f%%\n" (results.total_return *. 100.);
    Printf.printf "Sharpe Ratio: %.2f\n" results.sharpe_ratio;
    Printf.printf "Max Drawdown: %.2f%%\n" (results.max_drawdown *. 100.)

  let run_distributed_simulation () =
    let num_simulations = ref 1
    and num_workers = ref 1 in
    
    let speclist = [
      ("--num_simulations", Arg.Set_int num_simulations, "Number of simulations to run");
      ("--num_workers", Arg.Set_int num_workers, "Number of worker threads");
    ] in
    
    Arg.parse speclist (fun _ -> ()) "Usage: distributed_simulation [options]";
    
    let config = User_interface.create_default_config () in
    let configs = List.init !num_simulations (fun _ -> config) in
    let results = User_interface.run_distributed_simulation configs !num_workers in
    
    Printf.printf "Distributed Simulation Results:\n";
    List.iteri (fun i simulation_results ->
      Printf.printf "Simulation %d:\n" (i + 1);
      Analysis.generate_report [simulation_results] (Printf.sprintf "Simulation %d" (i + 1))
    ) results
end