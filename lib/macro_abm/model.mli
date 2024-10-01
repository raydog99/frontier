open Sector
open Firm
open Household
open Market
open Housing_market
open Financial_market
open Government
open Central_bank

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

val create : int -> int -> int -> int -> t
val calculate_aggregate_demand : t -> float
val calculate_aggregate_supply : t -> float
val update_inflation : t -> unit
val update_gdp : t -> unit
val update_unemployment : t -> unit
val housing_market_step : t -> unit
val financial_market_step : t -> unit
val credit_market_step : t -> unit
val step : t -> unit