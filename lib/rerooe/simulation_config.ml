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

let create ~model_type ~strategy ~optimizer ~market_impact ~transaction_costs ~params ~x0 ~s0 ~num_paths ~dt ~num_steps =
  { model_type; strategy; optimizer; market_impact; transaction_costs; params; x0; s0; num_paths; dt; num_steps }

let update_params config new_params =
  { config with params = new_params }

let update_strategy config new_strategy =
  { config with strategy = new_strategy }