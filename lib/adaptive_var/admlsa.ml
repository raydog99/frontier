open Torch
open Lwt.Infix
open Types
open Config
open Utils
open Logging
open Framework
open Loss_functions
open Adaptive_refinement
open Adaptive_strategy
open Error_analysis
open Var_estimation
open Random_generator
open Profiler
open Stopping_criterion

let admlsa_level config framework phi eps l benchmark adaptive_strategy cache logger profiler =
  let cache_key = Printf.sprintf "%f_%d" eps l in
  match Cache.get cache cache_key with
  | Some cached_result ->
      logger.log Info (Printf.sprintf "Using cached result for level %d" l);
      Lwt.return cached_result
  | None ->
      let module Gen = (val config.random_generator) in
      let rec compute_level_result n acc =
        if n = 0 then Lwt.return acc
        else
          let y = Tensor.of_float1 [|Gen.normal ()|] in
          let z = Tensor.of_float1 (Array.init (config.m ** l) (fun _ -> Gen.normal ())) in
          let x_refined = refine config framework phi y z (Tensor.to_float0_exn acc) n l in
          let h_xi = indicator_function (x_refined - acc) in
          let step = step_size config n in
          let new_acc = Tensor.(acc - (f step * (f config.alpha - h_xi))) in
          compute_level_result (n - 1) new_acc
      in
      let n = optimal_iterations config eps framework l in
      let initial_value = Tensor.of_float0 (Gen.normal ()) in
      compute_level_result n initial_value >>= fun result ->
      let performance = 0.5 in
      let convergence_rate = 1.0 in
      let error = 0.1 in
      update adaptive_strategy ~performance ~convergence_rate ~error ~level:l;
      Cache.add cache cache_key result;
      Lwt.return result

let admlsa config framework phi eps l =
  let logger = create_file_logger "admlsa.log" in
  logger.log Info (Printf.sprintf "Starting ADMLSA with eps=%f, l=%d" eps l);
  
  let config =
    if config.auto_tune_hyperparameters then
      let optimal_params, _ = Hyperparameter_tuning.bayesian_optimization config framework phi eps l in
      logger.log Info (Printf.sprintf "Optimal hyperparameters: theta=%f, r=%f, ca=%f, m=%d"
        optimal_params.theta optimal_params.r optimal_params.ca optimal_params.m);
      { config with
        theta = optimal_params.theta;
        r = optimal_params.r;
        ca = optimal_params.ca;
        m = optimal_params.m;
      }
    else
      config
  in
  
  let benchmark = if config.benchmark then Some (Benchmark.create ()) else None in
  let adaptive_strategy = create ~theta:config.theta ~r:config.r ~ca:config.ca ~m:config.m in
  let cache = Cache.create 100 in
  let profiler = if config.profiling_enabled then create () else create () in
  let stopping_criterion = create ~window_size:config.stopping_window_size ~tolerance:config.stopping_tolerance in
  
  (match benchmark with Some b -> Benchmark.mark b "ADMLSA start" | None -> ());
  if config.profiling_enabled then start profiler "ADMLSA Total";
  
  let rec run_levels current_l acc =
    if current_l > l || should_stop stopping_criterion acc then
      Lwt.return acc
    else
      let compute_level level =
        admlsa_level config framework phi eps level benchmark adaptive_strategy cache logger profiler
      in
      (if config.use_distributed then
        let cluster = Distributed.create_cluster (Option.get config.master_address) config.slave_addresses in
        let distributed_work = Distributed.distribute_work cluster [current_l] in
        Distributed.collect_results distributed_work
      else
        Parallel.parallel_map_chunked ~chunks:config.parallel_chunks ~f:compute_level [current_l])
      >>= fun level_results ->
      let level_result = List.hd level_results in
      let new_acc = acc +. level_result in
      run_levels (current_l + 1) new_acc
  in
  
  Lwt.catch
    (fun () ->
      run_levels 0 0. >>= fun result ->
      if config.profiling_enabled then stop profiler;
      (match benchmark with Some b -> Benchmark.stop b; Benchmark.report b | None -> Lwt.return_unit) >>= fun () ->
      let true_value = Tensor.full [1] 0.5 in
      let estimated_value = Tensor.full [1] result in
      let error_stats = compute_error_stats true_value estimated_value in
      let (ci_lower, ci_upper) = confidence_interval result error_stats config.confidence_level in
      logger.log Info (Printf.sprintf "ADMLSA completed. Result: %f" result);
      logger.log Info (Printf.sprintf "Confidence Interval (%d%%): [%f, %f]" 
        (int_of_float (config.confidence_level *. 100.)) ci_lower ci_upper);
      logger.log Info (Printf.sprintf "Relative Error: %f" (relative_error (Tensor.to_float0_exn true_value) result));
      if config.profiling_enabled then print profiler;
      Lwt.return result)
    (function
     | Error e as exc ->
         logger.log Error (Error.to_string e);
         if config.profiling_enabled then stop profiler;
         Lwt.fail exc
     | e ->
         logger.log Error (Printexc.to_string e);
         if config.profiling_enabled then stop profiler;
         Lwt.fail (Error (ComputationError (Printexc.to_string e))))