open Torch

module Parameters : sig
  type system_params = {
    coupling_strength: float;
    noise_level: float option;
    initial_conditions: float array option;
  }
  
  type simulation_params = {
    max_dim: int;
    r: float;
    k_neighbors: int;
  }
  
  type optimization_params = {
    n_particles: int;
    max_iterations: int;
    w: float;
    c1: float;
    c2: float;
  }
  
  type coupling_params = {
    bins: int;
    k_neighbors: int;
  }
end

type time_series = Tensor.t

type embedding_vector = {
  dimensions: int array;  (** m_i for each variable *)
  delays: int array;      (** τ_i for each variable *)
}

type prediction_error = {
  nrmse: float;
  component_errors: float array;
}

val scale_to_unit_interval : time_series -> time_series
val standardize : time_series -> time_series
val scale_multivariate : time_series array -> method_:[`UnitInterval | `Standardize] -> time_series array
val train_test_split : time_series array -> train_ratio:float -> time_series array * time_series array

val validate_preprocessing : time_series array -> unit
val embed_univariate : time_series -> dim:int -> delay:int -> time_series
val embed_multivariate : time_series array -> embedding_vector -> time_series

val validate_embedding_params : embedding_vector -> int -> unit
val validate_embedded_data : time_series -> unit

val calc_fnn : time_series -> float -> float
val find_optimal : time_series array -> max_dim:int -> float -> embedding_vector * float
val validate_fnn_params : int -> float -> unit
val generate_embeddings : int -> int -> embedding_vector list
val calc_fnn_increase : time_series array -> embedding_vector -> int -> float -> float
val find_optimal : time_series array -> max_dim:int -> float -> embedding_vector * float
val calculate_nrmse : Tensor.t -> Tensor.t -> prediction_error
val optimize_embedding : 
  time_series array -> 
  max_dim:int -> 
  k_neighbors:int -> 
  embedding_vector * float
val correlation_matrix : time_series array -> Tensor.t
val detect_redundancy : time_series array -> float -> bool array
val filter_redundant : time_series array -> float -> time_series array

val generate_ikeda : int -> ?params:Parameters.system_params -> unit -> time_series array
val generate_henon : int -> ?params:Parameters.system_params -> unit -> time_series array
val generate_kdr : int -> ?params:Parameters.system_params -> unit -> time_series array
val generate_coupled_henon : 
  int -> 
  params:Parameters.system_params -> 
  lattice_size:int -> 
  lattice_topology:lattice_topology -> 
  time_series array

module Models : sig
  type model_type = ZeroOrder | LinearModel | WeightedLinear
  
  val find_neighbors_radius : Tensor.t -> Tensor.t -> float -> Tensor.t
  val find_neighbors_knn : Tensor.t -> Tensor.t -> int -> Tensor.t
  
  val predict : 
    Tensor.t ->          (* embedded data *)
    Tensor.t ->          (* target point *)
    Tensor.t ->          (* target values *)
    ?model_type:model_type ->
    ?radius:float option ->
    ?k_neighbors:int option ->
    Tensor.t             (* predictions *)
    
  val validate_model_params : model_type -> float option -> int option -> unit
end

module MonteCarlo : sig
  type simulation_result = {
    embedding: embedding_vector;
    nrmse: float;
    frequency: int;
  }
  
  val run_simulation : 
    system_gen:(unit -> time_series array) ->
    n_trials:int ->
    methods:[`FNN1 | `FNN2 | `PEM] list ->
    params:Parameters.simulation_params ->
    (([`FNN1 | `FNN2 | `PEM] * simulation_result) list) list
    
  val check_termination_criteria : float list -> float -> int -> bool
end

module Topology : sig
  type lattice_topology = Linear | Ring | Grid of int
  
  val get_neighbors : lattice_topology -> int -> int -> int list
  val validate_topology : lattice_topology -> int -> unit
end

module CouplingAnalysis : sig
  type coupling_metric = {
    mutual_info: float;
    correlation: float;
    prediction_error: float;
  }
  
  val mutual_information : time_series -> time_series -> int -> float
  val analyze_coupling : time_series -> time_series -> params:Parameters.coupling_params -> coupling_metric
  val analyze_lattice_coupling : time_series array -> lattice_topology -> Tensor.t
end

module Statistics : sig
  type summary_stats = {
    mean: float;
    std_dev: float;
    confidence_interval: float * float;
  }
  
  val summarize_results : MonteCarlo.simulation_result array -> summary_stats
  val friedman_test : ('a * float) list list -> float
  val confidence_interval : float array -> float -> float * (float * float)
end

module Optimization : sig
  type particle = {
    position: embedding_vector;
    velocity: float array;
    best_position: embedding_vector;
    best_score: float;
  }
  
  val optimize_pso : 
    time_series array -> 
    params:Parameters.optimization_params ->
    embedding_vector * float
end