open Torch

module Error : sig
  type t = 
    | ConfigError of string
    | OptimizationError of string
    | GPError of string
    | DataError of string
    | SubspaceError of string
    | ResourceError of string

  val to_string : t -> string
  val raise_error : t -> 'a
end

module Config : sig
  type gp_config = {
    noise_variance: float;
    length_scale: float;
    signal_variance: float;
  }

  type optimization_config = {
    max_iterations: int;
    batch_size: int;
    tolerance: float;
    patience: int;
  }

  type subspace_config = {
    initial_dim: int;
    max_dim: int;
    min_explained_variance: float;
  }

  type t = {
    input_dim: int;
    n_initial_points: int;
    gp: gp_config;
    optimization: optimization_config;
    subspace: subspace_config;
    random_seed: int option;
  }

  val create : 
    input_dim:int -> 
    ?n_initial_points:int ->
    ?gp_config:gp_config ->
    ?opt_config:optimization_config ->
    ?subspace_config:subspace_config ->
    ?random_seed:int ->
    unit -> t

  val validate : t -> (t, Error.t) result
end

module Dataset : sig
  type point = Tensor.t
  type value = float

  type t = {
    points: point array;
    values: value array;
    best_value: value;
    best_point: point;
  }

  val create : point array -> value array -> t
  val add : t -> point -> value -> t
  val add_batch : t -> point array -> value array -> t
  val size : t -> int
  val get_best : t -> point * value
  val to_tensor : t -> Tensor.t * Tensor.t
  val of_tensor : Tensor.t -> Tensor.t -> t
end

module GP : sig
  type t 
  type params = Config.gp_config

  val create : params -> Dataset.t -> t
  val predict : t -> Dataset.point -> float * float
  val update : t -> Dataset.t -> t
  val optimize_hyperparams : t -> Dataset.t -> t
  val get_gradients : t -> Dataset.t -> Tensor.t
end

module LineOpt : sig
  type line = {
    start: Dataset.point;
    direction: Dataset.point;
    length: float;
  }

  type acquisition = Dataset.point -> float

  val create : Dataset.point -> Dataset.point -> float -> line
  val sample_points : line -> int -> Dataset.point array
  val optimize : line -> acquisition array -> Config.optimization_config -> Dataset.point
end

module Subspace : sig
  type t = {
    dim: int;
    projection: Tensor.t;
    reconstruction: Tensor.t;
    explained_variance: float;
  }

  val create : Config.subspace_config -> t
  val project : t -> Dataset.point -> Dataset.point
  val reconstruct : t -> Dataset.point -> Dataset.point
  val update : t -> GP.t -> Dataset.t -> t
  val should_expand : t -> Dataset.t -> bool
  val expand : t -> t
end

(* Multi-armed bandit *)
module MAB : sig
  type arm = {
    id: int;
    rewards: float list;
    pulls: int;
  }

  type t = {
    arms: arm array;
    exploration_factor: float;
  }

  val create : int -> float -> t
  val select : t -> int
  val update : t -> int -> float -> t
end

module Monitor : sig
  type metric = {
    iteration: int;
    best_value: float;
    gp_likelihood: float;
    subspace_dim: int;
    runtime: float;
  }

  type history = metric list

  val create_metric : int -> float -> float -> int -> float -> metric
  val update_history : history -> metric -> history
  val check_convergence : history -> Config.optimization_config -> bool
  val save : string -> history -> unit
  val load : string -> history
end

module BOIDS : sig
  type result = {
    dataset: Dataset.t;
    history: Monitor.history;
    final_model: GP.t;
    runtime: float;
  }

  val optimize : 
    (Dataset.point -> float) -> 
    Config.t -> 
    result

  val continue_optimization :
    result ->
    Config.t ->
    result
end

module Benchmarks : sig
  type problem = {
    name: string;
    dim: int;
    bounds: float * float;
    evaluate: Dataset.point -> float;
    optimum: float option;
  }

  val create_suite : int -> problem array
  val load_real_world : string -> problem
  val evaluate_optimizer : 
    problem -> 
    Config.t -> 
    int -> 
    (float * float * float) (* mean, std, time *)
end