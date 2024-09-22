open Torch
open Market_analysis
open Portfolio_analysis
open Parallel_processing

module type Backtesting = sig
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
end

module Backtesting = struct
  type portfolio_strategy = float array -> float array
  type backtest_result = {
    portfolio_values: float array;
    sharpe_ratio: float;
    maximum_drawdown: float;
    total_return: float;
    annualized_return: float;
    annualized_volatility: float;
  }

  let backtest market_caps dates strategy transaction_cost rebalancing_frequency risk_free_rate =
    let n = Array.length market_caps in
    let m = Array.length market_caps.(0) in
    let portfolio_values = Array.make m 1. in
    let current_weights = Array.make n (1. /. float_of_int n) in
    let returns = Array.make (m - 1) 0. in
    
    for t = 1 to m - 1 do
      let market_weights_t0 = calculate_market_weights market_caps.(t-1) in
      let market_weights_t1 = calculate_market_weights market_caps.(t) in
      
      (* Update portfolio weights due to market movements *)
      Array.iteri (fun i w -> 
        current_weights.(i) <- w *. market_weights_t1.(i) /. market_weights_t0.(i)
      ) current_weights;
      
      (* Rebalance if necessary *)
      if t mod rebalancing_frequency = 0 then begin
        let new_weights = strategy market_weights_t1 in
        let turnover = Array.fold_left (fun acc i -> 
          acc +. abs (new_weights.(i) -. current_weights.(i))
        ) 0. (Array.init n (fun i -> i)) in
        let cost = turnover *. transaction_cost in
        portfolio_values.(t) <- portfolio_values.(t-1) *. (1. -. cost);
        current_weights <- new_weights
      end else begin
        portfolio_values.(t) <- portfolio_values.(t-1)
      end;
      
      (* Calculate portfolio return *)
      let portfolio_return = calculate_portfolio_return 
        market_weights_t0 market_weights_t1 current_weights in
      portfolio_values.(t) <- portfolio_values.(t) *. (1. +. portfolio_return);
      returns.(t-1) <- portfolio_return
    done;
    
    let sharpe_ratio = calculate_sharpe_ratio returns risk_free_rate in
    let maximum_drawdown = calculate_maximum_drawdown portfolio_values in
    let total_return = portfolio_values.(m-1) /. portfolio_values.(0) -. 1. in
    let years = float_of_int (m - 1) /. 252. in (* Assuming 252 trading days per year *)
    let annualized_return = (total_return +. 1.) ** (1. /. years) -. 1. in
    let annualized_volatility = 
      sqrt (Array.fold_left (fun acc r -> acc +. r *. r) 0. returns /. float_of_int (m - 1)) *. sqrt 252. in

    {
      portfolio_values;
      sharpe_ratio;
      maximum_drawdown;
      total_return;
      annualized_return;
      annualized_volatility;
    }

  let compare_strategies market_caps dates strategies transaction_cost rebalancing_frequency risk_free_rate =
    List.map (fun (name, strategy) ->
      let result = backtest market_caps dates strategy transaction_cost rebalancing_frequency risk_free_rate in
      (name, result)
    ) strategies

  let run_parameter_sweep market_caps dates strategy_gen transaction_costs rebalancing_frequencies risk_free_rate =
    List.flatten (
      List.map (fun tc ->
        List.map (fun rf ->
          let strategy = strategy_gen tc in
          let result = backtest market_caps dates strategy tc rf risk_free_rate in
          (tc, rf, result)
        ) rebalancing_frequencies
      ) transaction_costs
    )

  let parallel_backtest market_caps dates strategies transaction_cost rebalancing_frequency risk_free_rate =
    ParallelProcessing.parallel_map (fun (name, strategy) ->
      let result = backtest market_caps dates strategy transaction_cost rebalancing_frequency risk_free_rate in
      (name, result)
    ) (Array.of_list strategies)
    |> Array.to_list

  let backtest_stylized_fact_strategies market_caps dates transaction_cost rebalancing_frequency risk_free_rate =
    let returns = Market_data.get_returns { market_caps; dates } in
    let returns_tensor = Tensor.of_float_array2 returns in
    let current_weights = calculate_market_weights (Array.get market_caps (Array.length market_caps - 1)) in
    let current_weights_tensor = Tensor.of_float_array current_weights in
    let strategies = [
      ("Diversity Trend", calculate_diversity_trend_portfolio);
      ("Volatility Responsive", calculate_volatility_responsive_portfolio);
      ("Rank Momentum", fun hist curr -> calculate_rank_momentum_portfolio hist 20 curr);
      ("Cross-sectional Momentum", calculate_cross_sectional_momentum_portfolio returns_tensor 20 current_weights_tensor);
      ("Factor Tilted", calculate_factor_tilted_portfolio returns_tensor (Tensor.randn [3; Array.length dates]) current_weights_tensor);
      ("Optimal Holding", calculate_optimal_holding_portfolio returns_tensor current_weights_tensor);
    ] in
    parallel_backtest market_caps dates strategies transaction_cost rebalancing_frequency risk_free_rate