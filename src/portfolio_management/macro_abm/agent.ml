type t = {
  id: int;
  mutable wealth: float;
}

let create id initial_wealth =
  { id; wealth = initial_wealth }

let update_wealth agent amount =
  agent.wealth <- agent.wealth +. amount