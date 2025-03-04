open Torch

(* Create a diagonal matrix from a vector *)
val diag : t -> t

(* Compute matrix logarithm *)
val matrix_log : t -> t

(* Compute matrix exponential *)
val matrix_exp : t -> t

(* Fisher transformation *)
val fisher_transform : t -> t

(* Inverse Fisher transformation *)
val inverse_fisher_transform : t -> t

(* Compute Kronecker product of two matrices *)
val kron : t -> t -> t

(* Univariate Volatility Modeling *)
module UnivariateVolatility : sig
  (* EGARCH model parameters *)
  type egarch_params = {
    omega: float;
    alpha: float;
    gamma: float;
    beta: float;
  }
  
  (* Forecast volatility using EGARCH model *)
  val egarch_forecast : egarch_params -> float -> float -> float
  
  (* Standardize returns using volatility *)
  val standardize : t -> t -> t
end

(* Factor Loadings Transformations *)
module FactorLoadings : sig
  (* Transform factor loadings to tau parameterization *)
  val to_tau : t -> t
  
  (* Transform tau parameterization back to factor loadings *)
  val from_tau : t -> t
  
  (* Compute Jacobian matrix of the transformation *)
  val jacobian : t -> t
end

(* Idiosyncratic Correlation Handling *)
module IdiosyncraticCorrelation : sig
  (* Block structure types *)
  type block_structure = 
    | Unrestricted
    | FullBlock
    | SparseBlock
    | DiagonalBlock
  
  (* Vectorize lower triangle of a matrix *)
  val vecl : t -> t
  
  (* Convert vectorized lower triangle back to symmetric matrix *)
  val vecl_to_matrix : t -> int -> t
  
  (* Apply generalized Fisher transformation to correlation matrix *)
  val gft : t -> t
  
  (* Inverse generalized Fisher transformation *)
  val inverse_gft : t -> t
  
  (* Create equicorrelation block matrix *)
  val create_equicorrelation_block : int -> float -> t
end

(* Distributions and Likelihood Computations *)
module Distributions : sig
  (* Convolution t-distribution types *)
  type convolution_t_type =
    | MultivariateTDist
    | ClusterTDist
    | HeterogeneousTDist
  
  (* Log-likelihood of standardized t-distribution *)
  val log_likelihood_t_dist : int -> t -> float
  
  (* Log-likelihood for Multivariate-t distribution *)
  val log_likelihood_multivariate_t : int -> t -> t -> t -> float
  
  (* Compute log-likelihood components for convolution-t distribution *)
  val log_likelihood_components_convolution_t : 
    t -> t -> t -> convolution_t_type -> int array -> int array -> float array
  
  (* Compute information matrix for Multivariate t-distribution *)
  val compute_mt_information_matrix : t -> t -> int -> t * t
end

(* Score-Driven Parameter Updates *)
module ScoreDriven : sig
  (* Core parameters for score-driven model *)
  type score_driven_params = {
    kappa: t;  (* Intercept *)
    beta: t;   (* Persistence *)
    alpha: t;  (* Score coefficient *)
  }
  
  (* One-step update in score-driven model *)
  val update : score_driven_params -> t -> t -> t
  
  (* Compute factor loading score *)
  val compute_factor_loading_score : 
    t -> t -> float -> float -> int -> t
  
  (* Compute score for idiosyncratic correlation *)
  val compute_idiosyncratic_corr_score : 
    t -> t -> convolution_t_type -> int -> int array -> t
  
  (* Apply Tikhonov regularization to score *)
  val tikhonov_regularized_score : 
    t -> t -> float -> t
end

(* Dynamic Factor Correlation Model *)
module DynamicFactorCorrelation : sig
  (* Full model parameters *)
  type model_params = {
    factor_corr_params: ScoreDriven.score_driven_params;
    factor_loading_params: ScoreDriven.score_driven_params array;
    idiosyncratic_corr_params: ScoreDriven.score_driven_params;
    
    factor_dist_type: Distributions.convolution_t_type;
    factor_df: int;
    
    returns_dist_type: Distributions.convolution_t_type;
    returns_df: int array;
    
    tikhonov_lambda: float array;
  }
  
  (* Evolving model state *)
  type model_state = {
    factor_corr: t;
    factor_loadings: t array;
    idiosyncratic_corr: t;
  }
  
  (* Initialize model state *)
  val init_state : int -> int -> block_structure -> model_state
  
  (* Compute idiosyncratic residuals *)
  val compute_idiosyncratic_residuals : 
    t -> t -> t array -> t
  
  (* Joint model update step *)
  val update_joint : 
    model_params -> model_state -> t -> t -> model_state
  
  (* Update idiosyncratic correlation *)
  val update_idiosyncratic_corr : 
    model_params -> model_state -> t -> t
end