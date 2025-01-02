open Sector

type t = {
  id: int;
  sector: Sector.t;
  mutable capital: float;
  mutable labor: float;
  mutable intermediate_inputs: (Sector.t * float) list;
  mutable production: float;
  mutable price: float;
  mutable inventory: float;
  mutable expected_demand: float;
  mutable target_production: float;
  mutable debt: float;
  mutable equity: float;
}

let create id sector initial_capital initial_equity =
  { id; sector; capital = initial_capital; labor = 0.; intermediate_inputs = [];
    production = 0.; price = 1.; inventory = 0.; expected_demand = 0.; 
    target_production = 0.; debt = 0.; equity = initial_equity }

let produce firm input_output_matrix =
  let min_ratio = ref (firm.capital /. (List.assoc firm.sector input_output_matrix |> List.assoc firm.sector)) in
  List.iter (fun (sector, required) ->
    let available = List.assoc sector firm.intermediate_inputs in
    let ratio = available /. required in
    if ratio < !min_ratio then min_ratio := ratio
  ) (List.assoc firm.sector input_output_matrix);
  firm.production <- !min_ratio *. firm.labor

let set_price firm inflation demand_pull_param cost_push_param =
  let demand_pull_inflation = demand_pull_param *. (firm.expected_demand -. firm.production) /. firm.production in
  let cost_push_inflation = cost_push_param *. (firm.price -. firm.price) /. firm.price in
  firm.price <- firm.price *. (1. +. inflation) *. 
                (1. +. demand_pull_inflation) *. 
                (1. +. cost_push_inflation)

let update_expected_demand firm actual_demand growth_rate firm_specific_param =
  let firm_specific_growth = firm_specific_param *. (firm.production -. actual_demand) /. actual_demand in
  firm.expected_demand <- actual_demand *. (1. +. growth_rate) *. (1. +. firm_specific_growth)

let set_target_production firm inventory_param labor_param intermediate_param capital_param =
  let target_inventory = inventory_param *. firm.production in
  let labor_constraint = firm.expected_demand +. labor_param *. (firm.labor -. firm.expected_demand) in
  let intermediate_constraint = 
    List.fold_left (fun acc (_, amount) -> 
      min acc (firm.expected_demand +. intermediate_param *. (amount -. firm.expected_demand))
    ) max_float firm.intermediate_inputs in
  let capital_constraint = firm.expected_demand +. capital_param *. (firm.capital -. firm.expected_demand) in
  firm.target_production <- min (firm.expected_demand +. target_inventory -. firm.inventory)
                                (min (min labor_constraint intermediate_constraint) capital_constraint)

let invest firm interest_rate =
  let desired_investment = max 0. (firm.target_production -. firm.capital) in
  let max_investment = (firm.equity /. (1. -. 0.6)) -. firm.debt in
  let actual_investment = min desired_investment max_investment in
  firm.capital <- firm.capital +. actual_investment;
  actual_investment

let request_loan firm amount interest_rate =
  firm.debt <- firm.debt +. amount;
  firm.capital <- firm.capital +. amount

let repay_loan firm amount =
  firm.debt <- max 0. (firm.debt -. amount)

let update_equity firm profit =
  firm.equity <- firm.equity +. profit