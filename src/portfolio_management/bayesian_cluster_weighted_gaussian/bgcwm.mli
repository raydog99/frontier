open Torch

type params = private {
  n: int;                  (** Number of observations *)
  p: int;                  (** Number of features *)
  k: int;                  (** Number of components *)
  pi: Tensor.t;      (** Mixing proportions *)
  alpha: Tensor.t;   (** Intercepts *)
  beta: Tensor.t;    (** Regression coefficients *)
  sigma_sq: Tensor.t; (** Error variances *)
  mu: Tensor.t;      (** Component means *)
  sigma: Tensor.t;   (** Component covariances *)
}

type prior_params = private {
  dir_alpha: float;        (** Dirichlet concentration *)
  bnb_a: float;           (** Beta-Negative-Binomial a *)
  bnb_a_pi: float;        (** Beta-Negative-Binomial a_π *)
  bnb_b_pi: float;        (** Beta-Negative-Binomial b_π *)
  f_nu_l: float;          (** F distribution ν_l *)
  f_nu_r: float;          (** F distribution ν_r *)
  alpha_var: float;       (** Intercept variance *)
  sigma_shape: float;     (** IG shape for error variance *)
  sigma_rate: float;      (** IG rate for error variance *)
  mu_mean: Tensor.t; (** Mean for μ *)
  psi_shape: float;       (** Gamma shape for graphical lasso *)
  psi_rate: float;        (** Gamma rate for graphical lasso *)
}

type mcmc_state = private {
  params: params;
  priors: prior_params;
  z: Tensor.t;       (** Component assignments *)
  log_likelihood: float;
  iteration: int;
}

module Distributions : sig
  val sample_dirichlet : float -> int -> Tensor.t
  (** [sample_dirichlet alpha k] samples from Dirichlet(α) distribution with k components *)

  val sample_inverse_gamma : float -> float -> int list -> Tensor.t
  (** [sample_inverse_gamma shape rate size] samples from InverseGamma(shape, rate) *)

  val sample_wishart : float -> Tensor.t -> Tensor.t
  (** [sample_wishart df scale] samples from Wishart(df, scale) distribution *)

  val sample_beta_negative_binomial : float -> float -> float -> int
  (** [sample_beta_negative_binomial a a_pi b_pi] samples from BetaNegativeBinomial distribution *)

  val sample_half_cauchy : float -> int list -> Tensor.t
  (** [sample_half_cauchy scale size] samples from Half-Cauchy(scale) distribution *)
end

module Numerical : sig
  exception NumericalError of string

  val safe_cholesky : Tensor.t -> Tensor.t
  (** [safe_cholesky mat] computes Cholesky decomposition with numerical safeguards *)

  val safe_inverse : Tensor.t -> Tensor.t
  (** [safe_inverse mat] computes matrix inverse with numerical safeguards *)

  val safe_log_det : Tensor.t -> float
  (** [safe_log_det mat] computes log determinant with numerical safeguards *)

  val log_sum_exp : Tensor.t -> Tensor.t
  (** [log_sum_exp x] computes log(sum(exp(x))) in a numerically stable way *)
end

module VariableSelection : sig
  type credible_region = {
    lower: Tensor.t;
    upper: Tensor.t;
  }

  val compute_credible_regions : mcmc_state list -> float -> credible_region
  (** [compute_credible_regions samples alpha] computes (1-α)% credible regions *)

  val select_variables : credible_region -> Tensor.t
  (** [select_variables regions] identifies significant variables *)
end

module BGCWM : sig
  type t 

  val create : n:int -> p:int -> k:int -> t
  (** [create ~n ~p ~k] creates a new BGCWM model *)

  val init_mcmc : x:Tensor.t -> y:Tensor.t -> mcmc_state
  (** [init_mcmc ~x ~y] initializes MCMC state *)

  val run_mcmc : 
    x:Tensor.t -> 
    y:Tensor.t -> 
    init_state:mcmc_state -> 
    n_iter:int -> 
    mcmc_state list
  (** [run_mcmc ~x ~y ~init_state ~n_iter] runs MCMC sampling *)

  val run_analysis :
    x:Tensor.t ->
    y:Tensor.t ->
    k_min:int ->
    k_max:int ->
    n_iter:int ->
    alpha:float ->
    mcmc_state list * Tensor.t * VariableSelection.credible_region
  (** [run_analysis ~x ~y ~k_min ~k_max ~n_iter ~alpha] runs complete analysis *)

  val summarize_results : mcmc_state list -> Tensor.t -> unit
  (** [summarize_results samples significant_vars] prints summary statistics *)

  val compute_log_likelihood : params -> x:Tensor.t -> y:Tensor.t -> float
  (** [compute_log_likelihood params ~x ~y] computes model log likelihood *)

  val check_convergence : mcmc_state list -> float * float
  (** [check_convergence samples] computes convergence diagnostics *)
end

module BatchProcessor : sig
  type batch = {
    x: Tensor.t;
    y: Tensor.t;
    start_idx: int;
    size: int;
  }

  val create_batches : 
    x:Tensor.t -> 
    y:Tensor.t -> 
    batch_size:int -> 
    batch array
  (** [create_batches ~x ~y ~batch_size] creates batches for processing *)

  val process_batch : 
    BGCWM.t -> 
    batch -> 
    (Tensor.t, string) result
  (** [process_batch model batch] processes a single batch *)
end