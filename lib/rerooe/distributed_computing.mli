type task =
  | Simulation of Simulation_config.t
  | Optimization of Portfolio_optimizer.optimization_method * Constrained_optimizer.t
  | BacktestTask of Event_driven_backtester.t

type result =
  | SimulationResult of (float * float * float) list
  | OptimizationResult of float array
  | BacktestResult of Portfolio.performance_summary

val distribute_tasks : task list -> int -> result list Lwt.t
val aggregate_results : result list -> (float * float * float) list list * float array list * Portfolio.performance_summary list
val log_message : string -> unit
val handle_task_error : task -> exn -> unit
val distribute_tasks_with_retry : task list -> int -> int -> result list Lwt.t