open Torch

(** State representation for ODE systems *)
module StateTypes : sig
  type system_state = {
    values: Tensor.t;      (** Current state vector *)
    dimension: int;        (** System dimension *)
    time: float;          (** Current time *)
    derivatives: Tensor.t option;  (** Optional cached derivatives *)
  }

  type time_mesh = {
    points: float array;   (** Mesh points *)
    fine_dt: float;       (** Fine timestep *)
    coarse_dt: float;     (** Coarse timestep *)
    slice_indices: int array;  (** Time slice boundaries *)
  }

  type solver_config = {
    order: int;           (** Method order *)
    method_name: string;  (** Method identifier *)
    adaptive: bool;       (** Adaptive stepping flag *)
    tol: float;          (** Error tolerance *)
  }

  type solution_history = {
    states: system_state array;
    times: float array;
    iteration: int;
    from_fine: bool;
  }
end

module Solver : sig
  type t

  val make_rk4 : f:(float -> Tensor.t -> Tensor.t) -> tol:float -> t
  (** Create RK4 solver *)

  val make_euler : f:(float -> Tensor.t -> Tensor.t) -> tol:float -> t
  (** Create Forward Euler solver *)

  val solve : t -> t0:float -> tf:float -> init_state:Tensor.t -> 
    Tensor.t array * float array
  (** Solve ODE over interval *)
end

module GP : sig
  type t
  type training_point = private {
    input: Tensor.t;
    output: Tensor.t;
    time: float;
  }

  val create : dimension:int -> t
  (** Create new GP *)

  val add_point : t -> input:Tensor.t -> output:Tensor.t -> time:float -> t
  (** Add training point *)

  val predict : t -> x:Tensor.t -> t:float -> Tensor.t * Tensor.t
  (** Predict at new point, returns (mean, variance) *)

  val optimize : t -> t
  (** Optimize hyperparameters *)
end

module VectorGP : sig
  type t

  val create : dimension:int -> t
  (** Create vector-valued GP *)

  val add_point : t -> input:Tensor.t -> output:Tensor.t -> time:float -> t
  (** Add training point *)

  val predict : t -> x:Tensor.t -> t:float -> Tensor.t * Tensor.t
  (** Predict at new point *)

  val optimize : t -> t
  (** Optimize hyperparameters *)
end

module Correction : sig
  type t = private {
    value: Tensor.t;
    input_state: Tensor.t;
    time: float;
    iteration: int;
    from_gp: bool;
    quality_score: float option;
  }

  module Database : sig
    type t

    val create : max_size:int -> dimension:int -> t
    (** Create correction database *)

    val add : t -> t -> t
    (** Add correction *)

    val get_nearby : t -> time:float -> window:float -> t array
    (** Get nearby corrections *)
  end
end

module LegacyData : sig
  type run_data = private {
    states: StateTypes.system_state array;
    corrections: Correction.t array;
    times: float array;
    run_id: int;
  }

  type t

  val create : dimension:int -> t
  (** Create legacy data store *)

  val add_run : t -> run_data -> t
  (** Add run data *)
end

module GParareal : sig
  type t
  type config = {
    t0: float;           (** Initial time *)
    tf: float;           (** Final time *)
    num_slices: int;     (** Number of parallel slices *)
    dimension: int;      (** System dimension *)
    tol: float;          (** Convergence tolerance *)
    max_iterations: int; (** Maximum iterations *)
    fine_points_per_slice: int;  (** Fine resolution *)
    coarse_points_per_slice: int;  (** Coarse resolution *)
  }

  val create : f:(float -> Tensor.t -> Tensor.t) -> 
              dimension:int -> 
              t0:float -> 
              tf:float -> 
              num_slices:int -> 
              tol:float -> 
              ?legacy_data:LegacyData.t -> 
              unit -> t
  (** Create GParareal solver *)

  val solve : t -> Tensor.t -> t * StateTypes.system_state array * int
  (** Solve ODE system, returns (final_state, solutions, iterations) *)

  val get_solution_history : t -> 
    StateTypes.solution_history array array
  (** Get full solution history *)
end