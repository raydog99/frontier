open Torch

(* Time conditions *)
module TimeConditions : sig
  type state = Pos | Neg
  
  val to_int : state -> int
  val of_int : int -> state
  
  type transition_matrix = {
    p11: float;
    p12: float;
    p21: float;
    p22: float;
  }
  
  val transition_prob : transition_matrix -> state -> state -> float
  val create_transition_matrix : float -> float -> transition_matrix
  val sample_next_state : transition_matrix -> state -> state
end

(* Asset returns *)
module AssetReturns : sig
  type return_parameters = {
    mu_s: float;
    sigma_s: float;
    rf: float;
    sigma_p: float;
    rho: float;
    nu_p: float;
  }
  
  type state_dependent_returns = {
    pos: return_parameters;
    neg: return_parameters;
  }
  
  val get_parameters : state_dependent_returns -> TimeConditions.state -> return_parameters
  
  type pe_process_params = {
    rho_p1: float;
    rho_p2: float;
    theta0: float;
    theta1: float;
    theta2: float;
  }
  
  val default_pe_process_params : unit -> pe_process_params
  val validate_smoothing_weights : pe_process_params -> pe_process_params
  val generate_true_pe_return : float -> float -> float -> float
  val apply_return_smoothing : pe_process_params -> float -> float -> float -> float
  val calculate_observed_pe_return : pe_process_params -> float -> float -> float -> float
  val update_expected_pe_return : pe_process_params -> float -> float -> float -> float
  val calculate_true_pe_return : float -> float -> float * float
  
  type pe_return_history = {
    true_returns: float array;
    observed_returns: float array;
    current_idx: int;
  }
  
  val init_pe_return_history : float -> int -> pe_return_history
  val update_pe_return_history : pe_return_history -> float -> float -> pe_return_history
  val get_previous_returns : pe_return_history -> int -> float * float
  
  val generate_returns_with_history : 
    state_dependent_returns -> 
    pe_process_params -> 
    float -> 
    pe_return_history -> 
    TimeConditions.state -> 
    float * float * float * float * float * pe_return_history
    
  val generate_returns : 
    state_dependent_returns -> 
    pe_process_params -> 
    float -> 
    float -> 
    TimeConditions.state -> 
    float * float * float * float * float
    
  val calculate_return_statistics : float array -> float * float * float
  val calculate_covariance : float array -> float array -> float
  val calculate_correlation : float array -> float array -> float
  val unsmooth_returns : pe_process_params -> float array -> float array
end

(* Capital flows *)
module CapitalFlows : sig
  type flow_parameters = {
    lambda_k: float;
    lambda_n: float;
    lambda_d: float;
    alpha: float;
  }
  
  type state_dependent_flows = {
    pos: flow_parameters;
    neg: flow_parameters;
  }
  
  val get_parameters : state_dependent_flows -> TimeConditions.state -> flow_parameters
end

(* Risk budget *)
module RiskBudget : sig
  type risk_weights = {
    theta_b: float;
    theta_s: float;
    theta_p: float;
    kappa: float;
    threshold: float;
  }
  
  val portfolio_risk_weight : risk_weights -> float -> float -> float -> float -> float -> float -> float
  val default_risk_weight : risk_weights -> float -> float -> float -> float -> float -> float -> float
  val risk_cost : risk_weights -> float -> float -> float
  val marginal_risk_cost : risk_weights -> float -> float -> [< `Bond | `PE | `Stock ] -> float
  val risk_adjusted_return : risk_weights -> float -> float -> float -> [< `Bond | `PE | `Stock ] -> float
  val is_within_budget : risk_weights -> float -> bool
  
  val max_allocation_within_budget : 
    risk_weights -> 
    float -> 
    float -> 
    float -> 
    float -> 
    [< `Bond | `PE | `Stock ] -> 
    float
end

(* Model parameters *)
module ModelParams : sig
  type t = {
    business_cycle: TimeConditions.transition_matrix;
    asset_returns: AssetReturns.state_dependent_returns;
    pe_process: AssetReturns.pe_process_params;
    capital_flows: CapitalFlows.state_dependent_flows;
    risk_budget: RiskBudget.risk_weights;
    gamma: float;
    epsilon_n: float;
    epsilon_s: float;
    n_bar: float;
    s_bar: float;
    terminal_time: int;
    time_step: float;
    use_smoothed_returns: bool;
  }
  
  val default : unit -> t
  val aggressive : unit -> t
  val conservative : unit -> t
  val naive : unit -> t
  val high_risk_charge : unit -> t
end

(* Gaussian Process *)
module GP : sig
  module Matrix : sig
    type t = float array array
    
    val create : int -> int -> float -> t
    val add : t -> t -> t
    val mul : t -> t -> t
    val mul_vec : t -> float array -> float array
    val transpose : t -> t
    val cholesky : t -> t
    val solve_cholesky : t -> float array -> float array
    val log_det_cholesky : t -> float
    val add_diagonal : t -> float -> t
  end
  
  val matern52_kernel : float array -> float array -> float array -> float -> float
  
  type gp_params = {
    mean: float;
    sigma_f: float;
    lengthscales: float array;
    sigma_n: float;
  }
  
  type t = {
    params: gp_params;
    x_train: float array array;
    y_train: float array;
    chol_k: float array array option;
    alpha: float array option;
    log_marginal_likelihood: float option;
  }
  
  val init : gp_params -> t
  val compute_kernel_matrix : t -> float array array -> float array array
  val calculate_log_marginal_likelihood : t -> float array array -> float array -> float * float array array * float array
  val fit : t -> float array array -> float array -> t
  val predict : t -> float array -> float
  val optimize_hyperparameters : t -> float array array -> float array -> t
end