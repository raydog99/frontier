open Torch

type t

val create : ?alpha:float -> ?max_iterations:int -> ?tolerance:float ->
             ?rel_tolerance:float -> ?stagnation_window:int ->
             ?parallel:bool -> ?logging_level:Logs.level -> ?adaptive_step:bool ->
             ?preconditioner:Tensor.t option -> ?early_stopping:bool ->
             ?devices:int list -> ?checkpoint_interval:int option ->
             ?checkpoint_path:string option -> ?plugins:(module Plugin.S) list ->
             unit -> t

type proximity_fn = Tensor.t -> Tensor.t
type constraint_fn = Tensor.t -> Tensor.t
type validation_fn = Tensor.t -> float

exception ConvergenceError of string
exception CheckpointError of string
exception PluginError of string

type iteration_result = {
  solution: Tensor.t;
  iterations: int;
  converged: bool;
  final_residual: float;
  convergence_history: (int * float) list;
  stopping_criterion: string;
  validation_history: (int * float) list option;
  performance_metrics: (string * float) list;
  plugin_results: (string * Yojson.Safe.t) list;
}

val solve : t -> ?prox_op:proximity_fn -> ?constraint_fn:constraint_fn -> 
            ?validation_fn:validation_fn -> (Tensor.t -> Tensor.t) -> Tensor.t -> iteration_result

val project_onto_ball : Tensor.t -> float -> Tensor.t -> Tensor.t
val project_onto_simplex : Tensor.t -> Tensor.t
val solve_lasso : ?lambda:float -> Tensor.t -> Tensor.t -> Tensor.t -> iteration_result
val solve_ridge : ?lambda:float -> Tensor.t -> Tensor.t -> Tensor.t -> iteration_result
val solve_elastic_net : ?alpha:float -> ?lambda:float -> Tensor.t -> Tensor.t -> Tensor.t -> iteration_result