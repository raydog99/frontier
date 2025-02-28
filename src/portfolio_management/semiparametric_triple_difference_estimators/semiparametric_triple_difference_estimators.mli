open Torch

type observation_panel = {
  g : int;    (* Group indicator (0 or 1) *)
  d : int;    (* Domain indicator (0 or 1) *)
  x : Tensor.t; (* Covariates *)
  y0 : float; (* Outcome at time 0 *)
  y1 : float; (* Outcome at time 1 *)
}

type observation_rc = {
  g : int;    (* Group indicator (0 or 1) *)
  d : int;    (* Domain indicator (0 or 1) *)
  t : int;    (* Time indicator (0 or 1) *)
  x : Tensor.t; (* Covariates *)
  y : float;  (* Outcome *)
}

(* Module signatures for the estimators *)
module type ESTIMATOR_PANEL = sig
  type t

  val create : ?config:string -> unit -> t
  val fit : t -> observation_panel array -> t
  val predict : t -> Tensor.t -> float
end

module type ESTIMATOR_RC = sig
  type t

  val create : ?config:string -> unit -> t
  val fit : t -> observation_rc array -> t
  val predict : t -> Tensor.t -> float
end

(* Module signature for propensity score estimators *)
module type PROPENSITY_ESTIMATOR = sig
  type t

  val create : ?config:string -> unit -> t
  val fit : t -> Tensor.t array -> int array -> t
  val predict : t -> Tensor.t -> float
end

(* Utility functions *)
module Utils : sig
  val mean_float : float array -> float
  val filter_observations : 'a array -> ('a -> bool) -> 'a array
  val filter_by_gd : observation_panel array -> int -> int -> observation_panel array
  val float_array_of_tensor : Tensor.t -> float array
  val mean_tensor : Tensor.t -> Tensor.t
  val empirical_mean : float array -> float
  val compute_delta_y : observation_panel array -> float array
  val extract_covariates : observation_panel array -> Tensor.t array
  val extract_outcomes_panel : observation_panel array -> (float * float) array
  val extract_outcomes_rc : observation_rc array -> float array
  val extract_gd : observation_panel array -> (int * int) array
  val extract_gdt : observation_rc array -> (int * int * int) array
  val normal_cdf : float -> float
end

(* Neural Network-based outcome regression estimators *)
module NeuralOutcomeEstimator : sig
  (* Panel data outcome regression model *)
  module Panel : sig
    type model = {
      net: Module.t;
      optimizer: Optimizer.t;
      learning_rate: float;
      batch_size: int;
      epochs: int;
    }
    
    val create : ?learning_rate:float -> ?batch_size:int -> ?epochs:int -> ?hidden_dim:int -> int -> model
    val train : model -> observation_panel array -> int -> int -> model
    val predict : model -> Tensor.t -> float
    val get_mu_estimator : (int * int, model) Hashtbl.t -> int -> int -> Tensor.t -> float
  end

  (* Repeated cross-sections outcome regression model *)
  module RC : sig
    type model = {
      net: Module.t;
      optimizer: Optimizer.t;
      learning_rate: float;
      batch_size: int;
      epochs: int;
    }
    
    val create : ?learning_rate:float -> ?batch_size:int -> ?epochs:int -> ?hidden_dim:int -> int -> model
    val train : model -> observation_rc array -> int -> int -> int -> model
    val predict : model -> Tensor.t -> float
    val get_mu_estimator : (int * int * int, model) Hashtbl.t -> int -> int -> int -> Tensor.t -> float
  end
end

(* Neural Network-based propensity score estimators *)
module NeuralPropensityEstimator : sig
  (* Panel data propensity score model *)
  module Panel : sig
    type model = {
      net: Module.t;
      optimizer: Optimizer.t;
      learning_rate: float;
      batch_size: int;
      epochs: int;
    }
    
    val create : ?learning_rate:float -> ?batch_size:int -> ?epochs:int -> ?hidden_dim:int -> int -> model
    val train : model -> observation_panel array -> int -> int -> model
    val predict : model -> Tensor.t -> float
    val get_pi_estimator : (int * int, model) Hashtbl.t -> int -> int -> Tensor.t -> float
  end

  (* Repeated cross-sections propensity score model *)
  module RC : sig
    type model = {
      net: Module.t;
      optimizer: Optimizer.t;
      learning_rate: float;
      batch_size: int;
      epochs: int;
    }
    
    val create : ?learning_rate:float -> ?batch_size:int -> ?epochs:int -> ?hidden_dim:int -> int -> model
    val train : model -> observation_rc array -> int -> int -> int -> model
    val predict : model -> Tensor.t -> float
    val get_pi_estimator : (int * int * int, model) Hashtbl.t -> int -> int -> int -> Tensor.t -> float
  end

  (* No compositional changes assumption version *)
  module RC_NoCompChanges : sig
    type model = Panel.model
    
    val train_pooled : model -> observation_rc array -> int -> int -> model
  end
end

(* Simple neural network-based estimators *)
module NNOutcomeEstimator : ESTIMATOR_PANEL
module NNPropensityEstimator : PROPENSITY_ESTIMATOR

(* Problem Setting implementation *)
module ProblemSetting : sig
  (* Panel Data Setting *)
  module PanelData : sig
    val compute_tau_p : observation_panel array -> float
    val generate_synthetic_data : ?n:int -> ?covariates_dim:int -> unit -> observation_panel array
  end

  (* Repeated Cross-Sections Setting *)
  module RepeatedCrossSections : sig
    val compute_tau_rc : observation_rc array -> float
    val generate_synthetic_data : ?n:int -> ?covariates_dim:int -> unit -> observation_rc array
  end
end

(* Identification implementation *)
module Identification : sig
  (* Panel data identification *)
  module PanelData : sig
    val identify_att_or : observation_panel array -> ((int * int) -> Tensor.t -> float) -> float
    val identify_att_ipw : observation_panel array -> (int -> int -> Tensor.t -> float) -> float
  end

  (* Repeated Cross-Sections identification *)
  module RepeatedCrossSections : sig
    val identify_att_or : observation_rc array -> ((int * int * int) -> Tensor.t -> float) -> float
    val identify_att_ipw : observation_rc array -> (int -> int -> int -> Tensor.t -> float) -> float
    val identify_att_ipw_no_comp_changes : observation_rc array -> (int -> int -> Tensor.t -> float) -> float
  end
end

(* Estimation implementation *)
module Estimation : sig
  (* Panel data estimation *)
  module PanelData : sig
    val estimate_att_dr : observation_panel array -> (int -> int -> Tensor.t -> float) -> (int -> int -> Tensor.t -> float) -> float
    val estimate_att_dr_crossfit : observation_panel array -> (observation_panel array -> int -> int -> Tensor.t -> float) -> (observation_panel array -> int -> int -> Tensor.t -> float) -> int -> float
  end

  (* Repeated Cross-Sections estimation *)
  module RepeatedCrossSections : sig
    val estimate_att_dr : observation_rc array -> (int -> int -> int -> Tensor.t -> float) -> (int -> int -> int -> Tensor.t -> float) -> float
    val estimate_att_dr_no_comp_changes : observation_rc array -> (int -> int -> int -> Tensor.t -> float) -> (int -> int -> Tensor.t -> float) -> float
    val estimate_att_dr_crossfit : observation_rc array -> (observation_rc array -> int -> int -> int -> Tensor.t -> float) -> (observation_rc array -> int -> int -> int -> Tensor.t -> float) -> int -> float
  end
end

(* Influence function implementations *)
module InfluenceFunctions : sig
  val if_att_or : observation_panel -> (int -> int -> Tensor.t -> float) -> float
  val if_att_ipw : observation_panel -> (int -> int -> Tensor.t -> float) -> float
  val if_att_dr : observation_panel -> (int -> int -> Tensor.t -> float) -> (int -> int -> Tensor.t -> float) -> float
end

(* Bootstrap confidence intervals *)
val bootstrap_ci : (observation_panel array -> float) -> observation_panel array -> int -> float -> float * float
val bootstrap_ci_rc : (observation_rc array -> float) -> observation_rc array -> int -> float -> float * float

(* Triple Difference estimators *)
module TripleDifference : sig
  (* Configuration type *)
  type config = {
    learning_rate: float;
    batch_size: int;
    epochs: int;
    hidden_dim: int;
    bootstrap_samples: int;
    alpha: float;
  }

  (* Default configuration *)
  val default_config : config

  (* Panel data triple difference estimator *)
  module PanelData : sig
    val train_outcome_models : ?config:config -> observation_panel array -> (int * int, NeuralOutcomeEstimator.Panel.model) Hashtbl.t
    val train_propensity_models : ?config:config -> observation_panel array -> (int * int, NeuralPropensityEstimator.Panel.model) Hashtbl.t
    
    val simple_diff : observation_panel array -> float
    val outcome_regression : ?config:config -> observation_panel array -> float
    val inverse_propensity_weighting : ?config:config -> observation_panel array -> float
    val doubly_robust : ?config:config -> observation_panel array -> float
    val doubly_robust_with_ci : ?config:config -> observation_panel array -> float * float * float
    val doubly_robust_crossfit : ?config:config -> ?k_folds:int -> observation_panel array -> float
  end

  (* Repeated cross-sections triple difference estimator *)
  module RepeatedCrossSections : sig
    val train_outcome_models : ?config:config -> observation_rc array -> (int * int * int, NeuralOutcomeEstimator.RC.model) Hashtbl.t
    val train_propensity_models : ?config:config -> observation_rc array -> (int * int * int, NeuralPropensityEstimator.RC.model) Hashtbl.t
    val train_propensity_models_no_comp_changes : ?config:config -> observation_rc array -> (int * int, NeuralPropensityEstimator.Panel.model) Hashtbl.t
    
    val simple_diff : observation_rc array -> float
    val outcome_regression : ?config:config -> observation_rc array -> float
    val inverse_propensity_weighting : ?config:config -> observation_rc array -> float
    val ipw_no_comp_changes : ?config:config -> observation_rc array -> float
    val doubly_robust : ?config:config -> observation_rc array -> float
    val doubly_robust_no_comp_changes : ?config:config -> observation_rc array -> float
    val doubly_robust_with_ci : ?config:config -> observation_rc array -> float * float * float
    val doubly_robust_crossfit : ?config:config -> ?k_folds:int -> observation_rc array -> float
  end
end

(* Monte Carlo simulation for evaluating the estimators *)
module MonteCarlo : sig
  (* Simulation parameters *)
  type sim_params = {
    n_units: int;
    dim_x: int;
    true_att: float;
    n_simulations: int;
    seed: int;
  }
  
  (* Default simulation parameters *)
  val default_params : sim_params
  
  (* Generate synthetic data for simulations *)
  val generate_panel_data : sim_params -> observation_panel array
  val generate_rc_data : sim_params -> observation_rc array
  
  (* Run simulations *)
  val run_panel_simulations : ?params:sim_params -> ?config:TripleDifference.config -> unit -> float array * float array * float array * float array
  val run_rc_simulations : ?params:sim_params -> ?config:TripleDifference.config -> ?assume_no_comp_changes:bool -> unit -> float array * float array * float array * float array * float array * float array
  
  (* Sensitivity analyses *)
  val sample_size_sensitivity : ?base_params:sim_params -> ?config:TripleDifference.config -> int array -> (int * float) array
  val effect_size_sensitivity : ?base_params:sim_params -> ?config:TripleDifference.config -> float array -> (float * float) array
end