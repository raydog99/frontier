type portfolio_strategy = float array -> float array
type backtest_result = {
    portfolio_values: float array;
    sharpe_ratio: float;
    maximum_drawdown: float;
    total_return: float;
    annualized_return: float;
    annualized_volatility: float;
  }

val backtest : 
    float array array ->  (* market_caps *)
    string array ->       (* dates *)
    portfolio_strategy -> (* strategy *)
    float ->              (* transaction_cost *)
    int ->                (* rebalancing_frequency *)
    float ->              (* risk_free_rate *)
    backtest_result

val compare_strategies :
    float array array ->  (* market_caps *)
    string array ->       (* dates *)
    (string * portfolio_strategy) list -> (* named strategies *)
    float ->              (* transaction_cost *)
    int ->                (* rebalancing_frequency *)
    float ->              (* risk_free_rate *)
    (string * backtest_result) list

val run_parameter_sweep :
    float array array ->  (* market_caps *)
    string array ->       (* dates *)
    (float -> portfolio_strategy) -> (* strategy generator *)
    float list ->         (* transaction_costs *)
    int list ->           (* rebalancing_frequencies *)
    float ->              (* risk_free_rate *)
    (float * int * backtest_result) list

val backtest_stylized_fact_strategies :
    float array array ->  (* market_caps *)
    string array ->       (* dates *)
    float ->              (* transaction_cost *)
    int ->                (* rebalancing_frequency *)
    float ->              (* risk_free_rate *)
    (string * backtest_result) list