open Torch

(* Configuration parameters for the robust optimization *)
type config = {
  n : int;                      (* Number of assets *)
  t : int;                      (* Number of periods *)
  confidence : float;           (* Confidence level (1 - delta_0) *)
  learning_rate : float;        (* Learning rate for optimization *)
  max_iter : int;               (* Maximum iterations for optimization *)
  rebalance_threshold : float;  (* Threshold for portfolio rebalancing *)
  transaction_cost : float;     (* Transaction cost as a percentage *)
  taylor_order : int;           (* Order of Taylor expansion (1 or 2) *)
}

(* Performance metrics for a portfolio strategy *)
type performance = {
  final_wealth : float;
  total_return : float;
  mean_daily_return : float;
  annualized_return : float;
  annualized_volatility : float;
  sharpe_ratio : float;
  max_drawdown : float;
  num_rebalances : int;
  total_transaction_costs : float;
}

(* Simulation results for a strategy *)
type simulation_result = {
  final_wealth : float;
  wealth_history : float array;
  rebalance_times : int list;
  total_transaction_costs : float;
  final_weights : Tensor.t;
}

(* Strategy representation *)
type strategy = {
  n : int;                              (* Number of assets *)
  periods : int;                        (* Number of periods *)
  strategy_fn : Tensor.t -> Tensor.t;   (* Function mapping returns to allocation *)
  threshold : float;                    (* Rebalancing threshold *)
  transaction_cost : float;             (* Transaction cost rate *)
}

(* Create default configuration for n assets and t periods *)
val default_config : n:int -> t:int -> config

(* Convert price data to returns *)
val prices_to_returns : Tensor.t -> Tensor.t

(* Load price data from a CSV file *)
val load_prices_from_csv : string -> string list * Tensor.t

(* Split data into training and testing sets *)
val split_train_test : Tensor.t -> float -> Tensor.t * Tensor.t

(* Generate synthetic price data for testing *)
val generate_synthetic_prices : n:int -> days:int -> volatility:float -> drift:float -> Tensor.t

(* Taylor series approximation for investment strategies *)
module TaylorApproximation : sig
  (* First-order Taylor approximation of the strategy *)
  val first_order_approximation : 
    initial_values:Tensor.t -> 
    gradients:Tensor.t -> 
    returns:Tensor.t -> 
    Tensor.t

  (* Second-order Taylor approximation of the strategy *)
  val second_order_approximation : 
    initial_values:Tensor.t -> 
    first_derivatives:Tensor.t -> 
    second_derivatives:Tensor.t -> 
    returns:Tensor.t -> 
    Tensor.t

  (* Create strategy function from coefficients *)
  val create_strategy_function : 
    coefficients:Tensor.t -> 
    t:int -> 
    n:int -> 
    order:int -> 
    Tensor.t -> Tensor.t
end

(* Robust Wasserstein Profile Inference for determining δ and α parameters *)
module RWPI : sig
  (* Find A* and lambda_0* by solving the non-robust problem *)
  val step1 : returns:Tensor.t -> Tensor.t * float

  (* Determine the Wasserstein ball radius δ *)
  val step2 : a_star:Tensor.t -> lambda_0:float -> returns:Tensor.t -> confidence:float -> float

  (* Determine the worst acceptable return α_bar *)
  val step3 : a_star:Tensor.t -> returns:Tensor.t -> delta:float -> confidence:float -> float

  (* Run the complete RWPI protocol *)
  val run : returns:Tensor.t -> confidence:float -> float * float * Tensor.t
end

(* Feasible region for the robust optimization problem *)
module FeasibleRegion : sig
  (* Check if a strategy is in the feasible region *)
  val check_feasibility : 
    coefficients:Tensor.t -> 
    returns:Tensor.t -> 
    delta:float -> 
    alpha_bar:float -> 
    bool

  (* Compute the worst-case expected return *)
  val worst_case_expected_return : 
    coefficients:Tensor.t -> 
    returns:Tensor.t -> 
    delta:float -> 
    float

  (* Project coefficients onto the feasible region *)
  val project_onto_feasible_region : 
    coefficients:Tensor.t -> 
    returns:Tensor.t -> 
    delta:float -> 
    alpha_bar:float -> 
    Tensor.t
end

(* Robust optimization solver for the dual problem *)
module RobustOptimizer : sig
  (* Dual objective function *)
  val dual_objective : 
    coefficients:Tensor.t -> 
    returns:Tensor.t -> 
    delta:float -> 
    float

  (* Solve the dual optimization problem *)
  val solve_dual_problem : 
    returns:Tensor.t -> 
    delta:float -> 
    alpha_bar:float -> 
    config:config -> 
    Tensor.t
end

(* Multi-period investment strategy *)
module MultiPeriodStrategy : sig
  (* Create a strategy *)
  val create : 
    n:int -> 
    periods:int -> 
    strategy_fn:(Tensor.t -> Tensor.t) -> 
    threshold:float -> 
    transaction_cost:float -> 
    strategy

  (* Compute target investment dividing portfolio into sub-portfolios *)
  val compute_target_investment : 
    strategy:strategy -> 
    returns:Tensor.t -> 
    time:int -> 
    Tensor.t

  (* Simulate a portfolio using the multi-period strategy with transaction costs *)
  val simulate : 
    strategy:strategy -> 
    initial_wealth:float -> 
    prices:Tensor.t -> 
    simulation_result
end

(* Calculate performance metrics from simulation results *)
val calculate_performance_metrics : 
  simulation_result:simulation_result -> 
  initial_wealth:float -> 
  performance

(* Robust mean-variance optimization *)
module RobustPortfolioOptimization : sig
  (* Train a multi-period robust strategy *)
  val train : 
    config:config -> 
    historical_returns:Tensor.t -> 
    strategy * float * float

  (* Train and evaluate a strategy *)
  val train_and_evaluate : 
    config:config -> 
    train_data:Tensor.t -> 
    test_data:Tensor.t -> 
    initial_wealth:float -> 
    strategy * performance * float * float

  (* Compare multiple strategies *)
  val compare_strategies : 
    strategies:(string * strategy) list -> 
    test_data:Tensor.t -> 
    initial_wealth:float -> 
    (string * performance) list
end

(* Analysis of the effect of number of periods *)
module PeriodAnalysis : sig
  (* Simulate with different numbers of periods *)
  val simulate_with_different_periods : 
    train_data:Tensor.t -> 
    test_data:Tensor.t -> 
    initial_wealth:float -> 
    period_counts:int list -> 
    (int * performance * float * float) list
    
  (* Split portfolio into n sub-portfolios starting at different times *)
  val create_split_portfolios : 
    strategy:(Tensor.t -> Tensor.t) -> 
    t:int -> 
    n:int -> 
    threshold:float -> 
    transaction_cost:float -> 
    strategy list
    
  (* Combine sub-portfolios to create a final multi-period strategy *)
  val combine_sub_portfolios : 
    sub_portfolios:strategy list -> 
    n:int -> 
    t:int -> 
    threshold:float -> 
    transaction_cost:float -> 
    strategy
end