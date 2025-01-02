open Torch
open Options_derivatives
open Real_time_data_pipeline

type stock = {
  ticker: string;
  returns: Tensor.t;
  market_cap: float;
  book_to_market: float;
  sentiment: float list;
  trading_volume: float list;
}

type portfolio = {
  stocks: stock list;
  weights: Tensor.t;
}

type factor_model = {
  factors: Tensor.t;
  factor_returns: Tensor.t;
  residuals: Tensor.t;
  r_squared: float;
}

type backtest_result = {
  portfolio_values: Tensor.t;
  returns: Tensor.t;
  benchmark_returns: Tensor.t;
  dates: string list;
  sharpe_ratio: float;
  information_ratio: float;
  max_drawdown: float;
  tail_index: float;
  cvar: float;
  final_portfolio: portfolio;
}

val create_stock : string -> Tensor.t -> float -> float -> float list -> float list -> stock
val create_portfolio : stock list -> Tensor.t -> portfolio
val calculate_portfolio_return : portfolio -> Tensor.t
val estimate_factor_model_gmm : Tensor.t -> Tensor.t -> int -> factor_model
val predict_returns_advanced : stock list -> Tensor.t -> Tensor.t
val construct_portfolio : stock list -> Tensor.t -> Tensor.t -> float -> portfolio
val update_portfolio_real_time : portfolio -> Real_time_data_pipeline.stock_data Queue.t -> portfolio
val advanced_backtest_parallel : 
  (portfolio -> Tensor.t -> string -> portfolio) -> 
  portfolio -> 
  Tensor.t -> 
  Tensor.t -> 
  float -> 
  float -> 
  float -> 
  Options_derivatives.option list -> 
  backtest_result