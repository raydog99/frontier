module Simulation_config : sig
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

  val create :
    model_type:[`Model1 | `Model2] ->
    strategy:Strategy.t ->
    optimizer:Optimizer.t ->
    market_impact:Market_impact.t ->
    transaction_costs:Transaction_costs.t ->
    params:Params.t ->
    x0:float ->
    s0:float ->
    num_paths:int ->
    dt:float ->
    num_steps:int ->
    t
end

module UserInterface : sig
  val create_default_config : unit -> Simulation_config.t
  val run_simulation : Simulation_config.t -> (float * float * float) list list
  val run_constrained_portfolio_optimization :
    Simulation_config.t -> (float * float * string) array -> float array ->
    Constrained_optimizer.constraint_type list -> Portfolio_optimizer.optimization_method ->
    float array
  val run_advanced_ml_strategy :
    Simulation_config.t -> Backtester.historical_data -> Ml.model -> Portfolio.performance_summary
  val run_distributed_simulation : Simulation_config.t list -> int -> (float * float * float) list list
  val run_and_visualize : Simulation_config.t -> unit
end

module CLI : sig
  val run : unit -> unit
  val run_advanced_ml_strategy : unit -> unit
  val run_distributed_simulation : unit -> unit
end