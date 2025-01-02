open Torch
open Lwt
open Logs
open Yojson.Basic.Util

open Ml
open Alternative_data_nlp
open Risk_modeling
open Parallel_processing
open Portfolio_construction
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

let create_stock ticker returns market_cap book_to_market sentiment trading_volume =
    { ticker; returns; market_cap; book_to_market; sentiment; trading_volume }

let create_portfolio stocks weights =
    { stocks; weights }

let calculate_portfolio_return portfolio =
    let stock_returns = List.map (fun s -> s.returns) portfolio.stocks in
    let returns_tensor = Tensor.stack stock_returns 0 in
    Tensor.mm returns_tensor (Tensor.unsqueeze portfolio.weights 1)
    |> Tensor.squeeze ~dim:[1]

let estimate_factor_model_gmm stock_returns factor_returns num_iterations =
    let betas = gmm_estimation stock_returns factor_returns num_iterations in
    let fitted_returns = Tensor.mm betas (Tensor.transpose factor_returns ~dim0:0 ~dim1:1) in
    let residuals = Tensor.sub stock_returns fitted_returns in
    
    let total_sum_squares = Tensor.sum (Tensor.pow stock_returns 2.) in
    let residual_sum_squares = Tensor.sum (Tensor.pow residuals 2.) in
    let r_squared = 1. -. (Tensor.to_float0_exn residual_sum_squares /. Tensor.to_float0_exn total_sum_squares) in
    
    { factors = betas; factor_returns; residuals; r_squared }

let predict_returns_advanced stocks historical_returns =
    let features = create_features stocks in
    
    let gbm_predictions = 
      parallel_map (fun features_row ->
        let gbm_model = GBM.train features historical_returns 100 5 0.1 in
        GBM.predict gbm_model features_row
      ) (Tensor.to_float2 features)
      |> Array.of_list
      |> Tensor.of_float1
    in
    
    let nn_model = NN.train features historical_returns [64; 32] 0.001 1000 in
    let nn_predictions = NN.predict nn_model features in
    
    let gpr_model = GPR.train features historical_returns 1.0 0.1 in
    let (gpr_predictions, _) = GPR.predict gpr_model features in
    
    Tensor.(div (add (add gbm_predictions nn_predictions) gpr_predictions) (scalar_tensor 3.))

let construct_portfolio stocks returns covariance_matrix risk_free_rate =
    let hrp_weights = hierarchical_risk_parity returns in
    let robust_mv_weights = robust_mean_variance_optimization returns covariance_matrix risk_free_rate 0.95 in
    let cdar_weights = cdar_optimization returns 0.05 0.95 in
    
    let combined_weights = Tensor.div (Tensor.add (Tensor.add (Tensor.of_float1 (Array.of_list hrp_weights)) robust_mv_weights) cdar_weights) (Tensor.scalar_tensor 3.) in
    create_portfolio stocks combined_weights

let update_portfolio_real_time portfolio data_queue =
    let update_stock stock =
      match Queue.find_opt (fun d -> d.ticker = stock.ticker) data_queue with
      | Some new_data ->
          let last_return = Tensor.get stock.returns [(Tensor.shape1_exn stock.returns) - 1] in
          let new_return = (new_data.price -. last_return) /. last_return in
          { stock with
            returns = Tensor.cat [stock.returns; Tensor.of_float0 new_return] ~dim:0;
            market_cap = new_data.price *. float_of_int new_data.volume;
            trading_volume = stock.trading_volume @ [float_of_int new_data.volume];
          }
      | None -> stock
    in
    let updated_stocks = List.map update_stock portfolio.stocks in
    let updated_returns = Tensor.stack (List.map (fun s -> s.returns) updated_stocks) 0 in
    let updated_weights = rebalance_portfolio updated_returns portfolio.weights in
    { stocks = updated_stocks; weights = updated_weights }

let advanced_backtest_parallel strategy initial_portfolio returns benchmark_returns risk_free_rate transaction_cost_rate risk_budget options =
    let num_periods = Tensor.shape1_exn returns in
    let chunk_size = 252 in
    let process_chunk start_idx end_idx =
      let chunk_returns = Tensor.slice returns ~dim:0 ~start:start_idx ~end_:end_idx in
      let chunk_benchmark_returns = Tensor.slice benchmark_returns ~dim:0 ~start:start_idx ~end_:end_idx in
      
      let rec simulate portfolio idx acc =
        if idx >= Tensor.shape1_exn chunk_returns then acc
        else
          let period_returns = Tensor.select chunk_returns ~dim:0 ~index:idx in
          let volatility = Tensor.std (Tensor.slice chunk_returns ~dim:0 ~start:(max 0 (idx-20)) ~end_:(idx+1)) ~dim:[0] ~unbiased:true in
          let macro_indicators = Tensor.randn [1; 5] in
          let market_regime = detect_market_regime period_returns volatility macro_indicators in
          
          let new_portfolio = strategy portfolio period_returns market_regime in
          let risk_adjusted_weights = risk_based_position_sizing new_portfolio.weights risk_budget (Tensor.std chunk_returns ~dim:[1] ~unbiased:true) in
          
          let hedged_weights = List.mapi (fun i w ->
            let stock = List.nth portfolio.stocks i in
            let stock_price = Tensor.select period_returns ~dim:0 ~index:i |> Tensor.to_float0_exn in
            let option = List.find_opt (fun o -> o.underlying = stock.ticker) options in
            match option with
            | Some o ->
              let hedge_amount = delta_hedge_strategy stock_price o 0.1 in
              w +. float_of_int hedge_amount /. 100.
            | None -> w
          ) (Tensor.to_float1 risk_adjusted_weights |> Array.to_list) in
          
          let hedged_weights_tensor = Tensor.of_float1 (Array.of_list hedged_weights) in
          let new_weights_with_costs = apply_transaction_costs hedged_weights_tensor portfolio.weights transaction_cost_rate in
          let period_return = Tensor.sum (Tensor.mul new_weights_with_costs period_returns) |> Tensor.to_float0_exn in
          
          let updated_portfolio = { new_portfolio with weights = new_weights_with_costs } in
          simulate updated_portfolio (idx + 1) ((period_return, updated_portfolio) :: acc)
      in
      simulate initial_portfolio 0 []
    in
    
    let chunk_results = 
      parallel_map (fun chunk_start ->
        let chunk_end = min (chunk_start + chunk_size) num_periods in
        process_chunk chunk_start chunk_end
      ) (List.init (num_periods / chunk_size + 1) (fun i -> i * chunk_size))
    in
    
    let all_results = List.flatten (List.rev chunk_results) in
    let portfolio_returns = Tensor.of_float1 (Array.of_list (List.map fst all_results)) in
    let final_portfolio = List.hd all_results |> snd in
    
    let sharpe = calculate_sharpe_ratio portfolio_returns risk_free_rate in
    let info_ratio = calculate_information_ratio portfolio_returns benchmark_returns in
    let max_dd = calculate_maximum_drawdown portfolio_returns in
    let (tail_index, cvar) = advanced_risk_model portfolio_returns in
    
    {
      portfolio_values = Tensor.cumprod (Tensor.add portfolio_returns 1.) ~dim:0;
      returns = portfolio_returns;
      benchmark_returns;
      dates = List.init num_periods (fun i -> Printf.sprintf "Day %d" (i + 1));
      sharpe_ratio = sharpe;
      information_ratio = info_ratio;
      max_drawdown = max_dd;
      tail_index;
      cvar;
      final_portfolio;
    }

let calculate_sharpe_ratio returns risk_free_rate =
    let excess_returns = Tensor.sub returns risk_free_rate in
    let mean_excess_return = Tensor.mean excess_returns in
    let std_dev = Tensor.std excess_returns ~dim:[0] ~unbiased:true in
    Tensor.div mean_excess_return std_dev |> Tensor.to_float0_exn

let calculate_information_ratio portfolio_returns benchmark_returns =
    let excess_returns = Tensor.sub portfolio_returns benchmark_returns in
    let mean_excess_return = Tensor.mean excess_returns in
    let tracking_error = Tensor.std excess_returns ~dim:[0] ~unbiased:true in
    Tensor.div mean_excess_return tracking_error |> Tensor.to_float0_exn

let calculate_maximum_drawdown returns =
    let cumulative_returns = Tensor.cumprod (Tensor.add returns 1.) ~dim:0 in
    let running_max = Tensor.cummax cumulative_returns ~dim:0 |> fst in
    let drawdowns = Tensor.div (Tensor.sub running_max cumulative_returns) running_max in
    Tensor.max drawdowns |> Tensor.to_float0_exn

let risk_based_position_sizing weights risk_budget individual_risks =
    let total_risk = Tensor.sum (Tensor.mul weights individual_risks) in
    let risk_contribution = Tensor.div (Tensor.mul weights individual_risks) total_risk in
    let scaling_factor = Tensor.div (Tensor.scalar_tensor risk_budget) risk_contribution in
    Tensor.mul weights scaling_factor

let apply_transaction_costs new_weights prev_weights transaction_cost_rate =
    let trades = Tensor.abs (Tensor.sub new_weights prev_weights) in
    let transaction_costs = Tensor.mul trades transaction_cost_rate in
    Tensor.sub new_weights transaction_costs

let detect_market_regime returns volatility macro_indicators =
    let features = Tensor.cat [returns; volatility; macro_indicators] ~dim:1 in
    let model = NN.train features (Tensor.of_float2 [|[|0.; 1.|]; [|1.; 0.|]|]) [32; 16] 0.001 1000 in
    let predictions = NN.predict model features in
    if Tensor.get predictions [0; 0] > Tensor.get predictions [0; 1] then
      "Low Volatility"
    else
      "High Volatility"