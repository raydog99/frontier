open Torch

(* Execution parameters capturing trading strategy constraints *)
type execution_params = {
  time_horizon: float;
  risk_aversion: float;
  price_sensitivity: Tensor.t;
  transaction_cost: Tensor.t;
  terminal_penalty: Tensor.t;
}

(* Market dynamics parameters *)
type market_params = {
  mean_reversion: Tensor.t;  (* R matrix *)
  volatility: Tensor.t;      (* V matrix *)
  long_term_mean: Tensor.t;  (* S_bar *)
  initial_price: Tensor.t;   (* S_0 *)
}

(* System state representation *)
type state = {
  time: float;
  inventory: Tensor.t;
  price: Tensor.t;
  cash: float;
}

(* Tensor operations module *)
module Tensor : sig
  val trace : Tensor.t -> Tensor.t
  val quadratic_form : Tensor.t -> Tensor.t -> Tensor.t -> float
  val symmetric_part : Tensor.t -> Tensor.t
end

(* Probability space and time discretization *)
module Filtration : sig
  type t = private {
    start_time: float;
    end_time: float;
    time_steps: int;
    dt: float;
  }

  val make_filtration : 
    start_time:float -> 
    end_time:float -> 
    time_steps:int -> 
    t

  val time_points : t -> float list
end

(* Stochastic process module *)
module Process : sig
  type 'a t = private {
    value: 'a;
    time: float;
    filtration: Filtration.t;
  }

  val create : 'a -> float -> Filtration.t -> 'a t
  val brownian_increment : dim:int -> float -> Tensor.t
  val generate_correlated_brownian : 
    Filtration.t -> int -> Tensor.t -> Tensor.t
end

(* Inventory management module *)
module Inventory : sig
  type t = private {
    quantity: Tensor.t;
    trading_rate: Tensor.t -> Tensor.t;
    process: Tensor.t Process.t;
  }

  val create : 
    initial_quantity:Tensor.t -> 
    trading_rate:(Tensor.t -> Tensor.t) -> 
    filtration:Filtration.t -> 
    t

  val evolve : t -> float -> Tensor.t
end

(* Price process module *)
module Price : sig
  type price_dynamics = {
    mean_reversion: Tensor.t;
    volatility: Tensor.t;
    long_term_mean: Tensor.t;
    correlation: Tensor.t;
  }

  type t = private {
    fundamental: Tensor.t Process.t;
    market: Tensor.t Process.t;
    dynamics: price_dynamics;
    price_sensitivity: Tensor.t;
  }

  val create : 
    initial_price:Tensor.t -> 
    dynamics:price_dynamics -> 
    price_sensitivity:Tensor.t -> 
    filtration:Filtration.t -> 
    t

  val evolve_fundamental : t -> float -> Tensor.t
  val evolve_market : t -> Tensor.t -> float -> Tensor.t Process.t
  val compute_covariance : price_dynamics -> Tensor.t
  val compute_drift : price_dynamics -> Tensor.t -> Tensor.t
end

(* Limit order book module *)
module LimitOrderBook : sig
  type t = private {
    balance: float;
    process: float Process.t;
    transaction_cost: Tensor.t;
  }

  val create : 
    initial_balance:float -> 
    transaction_cost:Tensor.t -> 
    filtration:Filtration.t -> 
    t

  val evolve : t -> Tensor.t -> Tensor.t -> float -> float
end

(* Utility function module *)
module UtilityFunction : sig
  type value_params = {
    a: Tensor.t;
    b: Tensor.t;
    c: Tensor.t;
    d: Tensor.t;
    e: Tensor.t;
    f: float;
  }

  val theta : value_params -> state -> float
  val utility : value_params -> state -> float
  val gradient : (state -> float) -> state -> Tensor.t
  val hessian : (state -> float) -> state -> Tensor.t
end

(* Hamilton-Jacobi-Bellman (HJB) module *)
module HJB : sig
  type hjb_params = {
    execution: execution_params;
    market: market_params;
  }

  val hamiltonian : 
    hjb_params -> state -> Tensor.t -> Tensor.t -> 
    Tensor.t * float * Tensor.t

  val optimal_control : 
    hjb_params -> state -> Tensor.t -> Tensor.t

  val hjb_rhs : 
    hjb_params -> state -> float -> Tensor.t -> Tensor.t -> Tensor.t

  val compute_diffusion_term : 
    (state -> float) -> state -> Tensor.t
end

(* Optimal execution module *)
module OptimalExecution : sig
  type execution_state = {
    time: float;
    inventory: Inventory.t;
    price: Price.t;
    cash: LimitOrderBook.t;
    utility_function: UtilityFunction.value_params;
  }

  val optimal_control : 
    hjb_params -> execution_state -> Tensor.t

  val solve : 
    hjb_params -> 
    initial_state:execution_state -> 
    execution_state
end

(* Solution interface module *)
module Solution : sig
  (* Error analysis components *)
  type error_components = {
    local_error: float;
    global_error: float;
    stability_factor: float;
  }

  (* Comprehensive solution result *)
  type solution_result = {
    final_state: OptimalExecution.execution_state;
    verification: Verification.verification_result;
    error_analysis: error_components;
  }

  (* Main solving function *)
  val solve : 
    hjb_params -> 
    initial_state:execution_state -> 
    (solution_result, string) result

  (* Parameter and state initialization *)
  val make_params : 
    time_horizon:float ->
    risk_aversion:float ->
    mean_reversion:Tensor.t ->
    volatility:Tensor.t ->
    long_term_mean:Tensor.t ->
    initial_price:Tensor.t ->
    price_sensitivity:Tensor.t ->
    transaction_cost:Tensor.t ->
    terminal_penalty:Tensor.t ->
    time_steps:int ->
    hjb_params

  val make_initial_state : 
    initial_inventory:Tensor.t ->
    initial_price:Tensor.t ->
    initial_cash:float ->
    filtration:Filtration.t ->
    OptimalExecution.execution_state
end