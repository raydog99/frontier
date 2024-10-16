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

val update_params : t -> Params.t -> t
val update_strategy : t -> Strategy.t -> t