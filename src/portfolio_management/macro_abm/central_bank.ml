type t = {
  mutable interest_rate: float;
  mutable money_supply: float;
}

let create initial_rate initial_money_supply =
  { interest_rate = initial_rate; money_supply = initial_money_supply }

let set_interest_rate bank inflation_rate target_inflation =
  bank.interest_rate <- max 0. (bank.interest_rate +. 0.5 *. (inflation_rate -. target_inflation))

let adjust_money_supply bank gdp_growth target_growth =
  bank.money_supply <- bank.money_supply *. (1. +. (gdp_growth -. target_growth))