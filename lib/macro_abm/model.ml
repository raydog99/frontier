open Torch
open Sector
open Firm
open Household
open Bank
open Market
open Housing_market
open Financial_market
open Government

type t = {
  sectors: Sector.t list;
  firms: Firm.t list;
  households: Household.t list;
  banks: Bank.t list;
  goods_markets: (Sector.t * Market.t) list;
  labor_market: Market.t;
  housing_market: Housing_market.t;
  financial_market: Financial_market.t;
  government: Government.t;
  central_bank: Central_bank.t;
  mutable inflation: float;
  mutable gdp: float;
  mutable unemployment_rate: float;
  input_output_matrix: (Sector.t * (Sector.t * float) list) list;
}

let create num_sectors num_firms num_households num_banks =
  let sectors = List.init num_sectors (fun i -> Sector.create i (Printf.sprintf "Sector_%d" i)) in
  let firms = List.init num_firms (fun i -> 
    Firm.create i (List.nth sectors (i mod num_sectors)) 100. 50.) in
  let households = List.init num_households (fun i -> Household.create i 1000. 10000.) in
  let banks = List.init num_banks (fun i -> Bank.create i 10000. 5000.) in
  let goods_markets = List.map (fun s -> (s, Market.create 1. 0.)) sectors in
  let labor_market = Market.create 1. 0. in
  let housing_market = Housing_market.create 100000. in
  let financial_market = Financial_market.create 100. in
  let government = Government.create 0.3 in
  let central_bank = Central_bank.create 0.02 1000000. in
  let input_output_matrix = 
    List.map (fun s -> 
      (s, List.map (fun s' -> (s', if s = s' then 0.5 else 0.1)) sectors)
    ) sectors in
  { sectors; firms; households; banks; goods_markets; labor_market; housing_market;
    financial_market; government; central_bank; inflation = 0.; gdp = 0.; 
    unemployment_rate = 0.; input_output_matrix }

let calculate_aggregate_demand model =
  List.fold_left (fun acc household -> acc +. household.Household.consumption) 0. model.households

let calculate_aggregate_supply model =
  List.fold_left (fun acc firm -> acc +. firm.Firm.production) 0. model.firms

let update_inflation model =
  let old_price = List.fold_left (fun acc (_, market) -> acc +. market.Market.price) 0. model.goods_markets in
  List.iter (fun (_, market) -> 
    Market.clear market (calculate_aggregate_demand model) (calculate_aggregate_supply model)
  ) model.goods_markets;
  let new_price = List.fold_left (fun acc (_, market) -> acc +. market.Market.price) 0. model.goods_markets in
  model.inflation <- (new_price -. old_price) /. old_price

let update_gdp model =
  model.gdp <- calculate_aggregate_supply model

let update_unemployment model =
  let total_labor_force = float_of_int (List.length model.households) in
  let employed = List.fold_left (fun acc firm -> acc +. firm.Firm.labor) 0. model.firms in
  model.unemployment_rate <- (total_labor_force -. employed) /. total_labor_force

let housing_market_step model =
  let demand = List.fold_left (fun acc h -> 
    if h.Household.income > model.housing_market.Housing_market.price *. 0.2 then acc +. 1. else acc
  ) 0. model.households in
  let supply = List.fold_left (fun acc h -> acc +. h.Household.housing) 0. model.households in
  Housing_market.update_price model.housing_market demand supply;
  List.iter (fun h -> 
    if h.Household.income > model.housing_market.Housing_market.price *. 0.2 && Random.float 1. < 0.1 then
      Household.buy_house h model.housing_market.Housing_market.price (model.central_bank.Central_bank.interest_rate +. 0.02)
  ) model.households;
  List.iter (fun h ->
    if h.Household.housing > 1. && Random.float 1. < 0.05 then
      Household.sell_house h model.housing_market.Housing_market.price
  ) model.households

let financial_market_step model =
  let demand = List.fold_left (fun acc h -> acc +. (0.1 *. h.Household.income)) 0. model.households in
  let supply = List.fold_left (fun acc f -> acc +. (0.1 *. f.Firm.equity)) 0. model.firms in
  Financial_market.update_price model.financial_market demand supply;
  List.iter (fun h -> 
    let investment = 0.1 *. h.Household.income in
    Household.invest h investment
  ) model.households

let credit_market_step model =
  List.iter (fun bank -> Bank.set_interest_rate bank model.central_bank.Central_bank.interest_rate) model.banks;
  List.iter (fun firm ->
    let loan_amount = 0.1 *. firm.Firm.expected_demand *. firm.Firm.price in
    let firm_leverage = firm.Firm.debt /. (firm.Firm.capital +. 0.001) in
    let bank = List.nth model.banks (Random.int (List.length model.banks)) in
    match Bank.process_loan bank loan_amount firm_leverage with
    | Some loan_rate -> Firm.request_loan firm loan_amount loan_rate
    | None -> ()
  ) model.firms;
  List.iter (fun h ->
    Household.pay_mortgage h (0.01 *. h.Household.mortgage)
  ) model.households

let step model =
  List.iter (fun firm -> Firm.produce firm model.input_output_matrix) model.firms;

  List.iter (fun firm -> Firm.set_price firm model.inflation 0.5 0.5) model.firms;

  let labor_demand = List.fold_left (fun acc firm -> acc +. firm.Firm.labor) 0. model.firms in
  let labor_supply = List.fold_left (fun acc household -> acc +. household.Household.labor_supply) 0. model.households in
  Market.clear model.labor_market labor_demand labor_supply;

  List.iter (fun h -> Household.update_reservation_wage h model.labor_market.Market.price) model.households;

  List.iter (fun household -> 
    let consumption_by_sector = Household.consume household (0.9 *. household.Household.income) model.sectors in
    List.iter (fun (sector, amount) ->
      let (_, market) = List.find (fun (s, _) -> s = sector) model.goods_markets in
      Market.clear market amount (calculate_aggregate_supply model)
    ) consumption_by_sector
  ) model.households;

  let total_income = List.fold_left (fun acc h -> acc +. h.Household.income) 0. model.households in
  let tax_revenue = Government.collect_taxes model.government total_income in
  Government.adjust_spending model.government model.gdp model.unemployment_rate;
  Government.issue_debt model.government;

  Central_bank.set_interest_rate model.central_bank model.inflation 0.02;
  Central_bank.adjust_money_supply model.central_bank ((model.gdp -. model.gdp) /. model.gdp) 0.03;

  housing_market_step model;

  financial_market_step model;

  credit_market_step model;

  update_inflation model;
  update_gdp model;
  update_unemployment model;

  let growth_rate = (model.gdp -. model.gdp) /. model.gdp in
  List.iter (fun firm -> Firm.update_expected_demand firm firm.Firm.production growth_rate 0.5) model.firms;

  List.iter (fun firm -> Firm.set_target_production firm 0.1 0.5 0.5 0.5) model.firms