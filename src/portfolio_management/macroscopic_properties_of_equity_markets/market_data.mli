module type MarketData = sig
  type t

  val load_data : string -> t
  val get_market_caps : t -> float array array
  val get_dates : t -> string array
  val get_returns : t -> float array array
  val get_log_returns : t -> float array array
  val filter_by_date_range : t -> string -> string -> t
  val filter_top_n_stocks : t -> int -> t
end

module CRSPData : MarketData