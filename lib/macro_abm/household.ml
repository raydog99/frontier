type t = {
  id: int;
  mutable income: float;
  mutable consumption: float;
  mutable labor_supply: float;
  mutable reservation_wage: float;
  mutable housing: float;
  mutable mortgage: float;
  mutable financial_assets: float;
}

let create id initial_income initial_financial_assets =
  { id; income = initial_income; consumption = 0.; labor_supply = 1.; 
    reservation_wage = 1.; housing = 0.; mortgage = 0.; financial_assets = initial_financial_assets }

let consume household amount sectors =
  household.consumption <- amount;
  List.map (fun s -> (s, amount /. float_of_int (List.length sectors))) sectors

let set_labor_supply household amount =
  household.labor_supply <- amount

let update_reservation_wage household market_wage =
  household.reservation_wage <- 0.9 *. household.reservation_wage +. 0.1 *. market_wage

let buy_house household price mortgage_rate =
  household.housing <- household.housing +. 1.;
  household.mortgage <- household.mortgage +. price;
  household.income <- household.income -. (mortgage_rate *. price)

let sell_house household price =
  household.housing <- max 0. (household.housing -. 1.);
  household.mortgage <- max 0. (household.mortgage -. price);
  household.income <- household.income +. price

let pay_mortgage household amount =
  household.mortgage <- max 0. (household.mortgage -. amount)

let decide_consumption household interest_rate =
  let disposable_income = household.income *. (1. -. 0.3) in
  let wealth_effect = 0.03 *. (household.housing *. 100000. +. household.financial_assets) in
  let consumption = 0.7 *. disposable_income +. 0.1 *. wealth_effect in
  household.consumption <- consumption

let decide_labor_supply household wage =
  let labor_supply = min 1. (max 0. (wage /. (household.reservation_wage *. 1.1))) in
  household.labor_supply <- labor_supply

let invest household amount =
  household.financial_assets <- household.financial_assets +. amount