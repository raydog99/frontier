open Lwt

type task =
  | Simulation of Simulation_config.t
  | Optimization of Portfolio_otimizer.optimization_method * Constrained_optimizer.t
  | BacktestTask of Event_driven_backtester.t

type result =
  | SimulationResult of (float * float * float) list
  | OptimizationResult of float array
  | BacktestResult of Portfolio.performance_summary

let log_message msg =
  let timestamp = Unix.gettimeofday () in
  Printf.printf "[%.6f] %s\n" timestamp msg

let handle_task_error task error =
  log_message (Printf.sprintf "Error in task: %s\nError: %s" 
    (match task with
     | Simulation _ -> "Simulation"
     | Optimization _ -> "Optimization"
     | BacktestTask _ -> "Backtest")
    (Printexc.to_string error))

let distribute_tasks tasks num_workers =
  let worker_pool = Lwt_pool.create num_workers (fun () -> Lwt.return_unit) in
  let process_task task =
    Lwt_pool.use worker_pool (fun () ->
      match task with
      | Simulation config ->
          let results = User_interface.run_simulation config in
          Lwt.return (SimulationResult results)
      | Optimization (method_, constraints) ->
          let config = User_interface.create_default_config () in
          let weights = User_interface.run_constrained_portfolio_optimization
            config [||] [||] constraints.Constrained_optimizer.constraints method_ in
          Lwt.return (OptimizationResult weights)
      | BacktestTask backtester ->
          Event_driven_backtester.run backtester (float_of_int (Array.length backtester.Event_driven_backtester.events - 1));
          let results = Event_driven_backtester.get_results backtester in
          Lwt.return (BacktestResult results)
    )
  in
  Lwt_list.map_p process_task tasks

let aggregate_results results =
  let simulations = ref []
  and optimizations = ref []
  and backtests = ref [] in
  List.iter (function
    | SimulationResult r -> simulations := r :: !simulations
    | OptimizationResult r -> optimizations := r :: !optimizations
    | BacktestResult r -> backtests := r :: !backtests
  ) results;
  (!simulations, !optimizations, !backtests)

let distribute_tasks_with_retry tasks num_workers max_retries =
  let rec retry_task task retries =
    if retries > max_retries then
      Lwt.fail (Failure "Max retries exceeded")
    else
      Lwt.catch
        (fun () -> 
          log_message (Printf.sprintf "Executing task (attempt %d)" (retries + 1));
          distribute_tasks [task] 1 >>= fun results ->
          Lwt.return (List.hd results))
        (fun error ->
          handle_task_error task error;
          log_message (Printf.sprintf "Retrying task (attempt %d)" (retries + 2));
          Lwt_unix.sleep (float_of_int retries) >>= fun () ->
          retry_task task (retries + 1))
  in
  Lwt_list.map_p (fun task -> retry_task task 0) tasks