module Logger : sig
  val log : string -> unit
  val close : unit -> unit
end

module Option_Type : sig
  type t =
    | GeometricBasketCall
    | MaxCall
    | StrangleSpreadBasket

  val payoff : t -> Option_Pricing.params -> Torch.Tensor.t -> Torch.Tensor.t
end

module Option_Pricing : sig
  type params = {
    strike : float;
    risk_free_rate : float;
    volatility : float;
    dividend_rate : float;
    option_type : Option_Type.t;
  }

  val simulate_paths_parallel :
    params ->
    Optimal_Stopping_Problem.problem_params ->
    int ->
    int ->
    Torch.Tensor.t Lwt.t

  val black_scholes_price :
    params -> Optimal_Stopping_Problem.problem_params -> float
end

module Optimal_Stopping_Problem : sig
  type problem_params = {
    dim : int;
    maturity : float;
    num_time_steps : int;
  }

  type t

  val create : problem_params -> Option_Pricing.params -> t

  val train : t -> int -> int -> int -> float -> unit Lwt.t

  val analyze_model :
    t ->
    int ->
    (float * float * Torch.Tensor.t * Torch.Tensor.t * float * float) Lwt.t
end

module Experiment : sig
  type experiment_params = {
    dim : int;
    maturity : float;
    num_time_steps : int;
    strike : float;
    risk_free_rate : float;
    volatility : float;
    dividend_rate : float;
    option_type : Option_Type.t;
    num_epochs : int;
    num_steps : int;
    batch_size : int;
    learning_rate : float;
    num_paths : int;
  }

  val run_experiment :
    experiment_params ->
    (float * float * Torch.Tensor.t * Torch.Tensor.t * float * float) Lwt.t

  val compare_experiments :
    experiment_params list ->
    (float * float * Torch.Tensor.t * Torch.Tensor.t * float * float) list Lwt.t
end

module Tests : sig
  val run_all_tests : unit -> unit
end