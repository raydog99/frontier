open Torch

(* Log gamma function *)
val log_gamma : Scalar.t -> Scalar.t

(* Digamma function (derivative of log gamma) *)
val digamma : Scalar.t -> Scalar.t

(* Transform constrained GARCH parameters to unconstrained space *)
val to_unconstrained : 
  omega:Tensor.t -> 
  alpha:Tensor.t -> 
  beta:Tensor.t -> 
  Tensor.t * Tensor.t * Tensor.t

(* Transform unconstrained parameters back to constrained GARCH space *)
val to_constrained : 
  omega_u:Tensor.t -> 
  psi1_u:Tensor.t -> 
  psi2_u:Tensor.t -> 
  Tensor.t * Tensor.t * Tensor.t

(* Transform degrees of freedom parameter *)
val transform_nu_to_unconstrained : Tensor.t -> Tensor.t
val transform_nu_from_unconstrained : Tensor.t -> Tensor.t

(* Transform skewness parameter *)
val transform_xi_to_unconstrained : Tensor.t -> Tensor.t
val transform_xi_from_unconstrained : Tensor.t -> Tensor.t

(* Compute GARCH(1,1) volatility process *)
val garch_1_1 : 
  omega:Tensor.t -> 
  alpha:Tensor.t -> 
  beta:Tensor.t -> 
  y:Tensor.t -> 
  Tensor.t

(* Compute partial derivatives of sigma2 with respect to unconstrained parameters *)
val d_sigma2_d_theta_garch_1_1 : 
  omega:Tensor.t ->
  alpha:Tensor.t ->
  beta:Tensor.t ->
  omega_u:Tensor.t ->
  psi1_u:Tensor.t ->
  psi2_u:Tensor.t ->
  y:Tensor.t ->
  Tensor.t

(* Priors for GARCH models *)
module Prior : sig
  type t = {
    log_pdf: Tensor.t -> Tensor.t;
    grad_log_pdf: Tensor.t -> Tensor.t;
  }
  
  (* Log-gamma prior for omega *)
  val neg_log_gamma : shape:Tensor.t -> rate:Tensor.t -> t
  
  (* Logistic prior for psi1 and psi2 *)
  val logistic : unit -> t
  
  (* Translated exponential prior for degrees of freedom *)
  val translated_exp : rate:Tensor.t -> t
  
  (* Inverse Gamma prior for skewness parameter *)
  val inverse_gamma : shape:Tensor.t -> rate:Tensor.t -> t
end

(* Gaussian GARCH model log-likelihood *)
val gaussian : y:Tensor.t -> sigma2:Tensor.t -> Tensor.t

(* Gradient of Gaussian GARCH log-likelihood w.r.t. unconstrained params *)
val grad_gaussian : y:Tensor.t -> sigma2:Tensor.t -> d_sigma2_d_theta:Tensor.t -> Tensor.t

(* Student's t GARCH model log-likelihood *)
val student_t : y:Tensor.t -> sigma2:Tensor.t -> nu:Tensor.t -> Tensor.t

(* Skew-t GARCH model log-likelihood *)
val skew_t : y:Tensor.t -> sigma2:Tensor.t -> nu:Tensor.t -> xi:Tensor.t -> Tensor.t

(* Gaussian Variational Approximation *)
module VariationalDist : sig
  type t = {
    mu: Tensor.t;
    l_chol: Tensor.t;
    d: int;
  }
  
  (* Create a variational distribution *)
  val create : d:int -> init_mu:Tensor.t -> init_l:Tensor.t -> t
  
  (* Sample from the variational distribution *)
  val sample : n:int -> t -> Tensor.t
  
  (* Compute log PDF of the variational distribution *)
  val log_pdf : t -> Tensor.t -> Tensor.t
  
  (* Compute entropy of the variational distribution *)
  val entropy : t -> Tensor.t
  
  (* Compute KL divergence between variational distribution and target *)
  val kl_divergence : q:t -> p_log_pdf:(Tensor.t -> Tensor.t) -> samples:Tensor.t -> Tensor.t
end

(* GARCH models module *)
module GarchModel : sig
  type innovation_type = Gaussian | StudentT | SkewT
  
  type t = {
    innovation_type: innovation_type;
    y: Tensor.t;
    priors: Tensor.t -> Tensor.t;
    grad_priors: Tensor.t -> Tensor.t;
  }
  
  (* Create a GARCH model *)
  val create : 
    innovation_type:innovation_type -> 
    y:Tensor.t -> 
    priors:(Tensor.t -> Tensor.t) -> 
    grad_priors:(Tensor.t -> Tensor.t) -> 
    t
  
  (* Compute GARCH volatility process *)
  val compute_sigma2 : t -> Tensor.t -> Tensor.t
  
  (* Compute log-likelihood for GARCH model *)
  val log_likelihood : t -> Tensor.t -> Tensor.t
end

(* Stochastic Variational Inference module *)
module SVI : sig
  type method_type = ControlVariates | Reparametrization
  
  type optimization_params = {
    learning_rate: float;
    beta1: float;  (* ADAM first moment decay *)
    beta2: float;  (* ADAM second moment decay *)
    max_iter: int;
    tol: float;
    patience: int;
    mc_samples: int;
    window_size: int;
  }
  
  (* Default optimization parameters *)
  val default_optimization_params : optimization_params
  
  (* ELBO computation for control variates approach *)
  val elbo_control_variates : 
    model:GarchModel.t -> 
    var_dist:VariationalDist.t -> 
    samples:Tensor.t -> 
    Tensor.t
  
  (* Stochastic gradient estimation with control variates *)
  val grad_elbo_cv : 
    model:GarchModel.t -> 
    var_dist:VariationalDist.t -> 
    samples:Tensor.t -> 
    Tensor.t
  
  (* ELBO computation for reparametrization approach *)
  val elbo_reparam : 
    model:GarchModel.t -> 
    var_dist:VariationalDist.t -> 
    eps:Tensor.t -> 
    mu:Tensor.t -> 
    l_chol:Tensor.t -> 
    Tensor.t
  
  (* Fit variational approximation using control variates *)
  val fit_cv : 
    model:GarchModel.t -> 
    init_var_dist:VariationalDist.t -> 
    opt_params:optimization_params -> 
    VariationalDist.t * Tensor.t list * int
  
  (* Fit variational approximation using reparametrization trick *)
  val fit_reparam : 
    model:GarchModel.t -> 
    init_var_dist:VariationalDist.t -> 
    opt_params:optimization_params -> 
    VariationalDist.t * Tensor.t list * int
end

(* Sequential variational inference module *)
module SequentialSVI : sig
  (* Updating Variational Bayes (UVB) *)
  val updating_vb : 
    model:GarchModel.t -> 
    init_var_dist:VariationalDist.t -> 
    opt_params:SVI.optimization_params -> 
    new_data:Tensor.t -> 
    VariationalDist.t * Tensor.t list * int
  
  (* Sequential Stochastic Variational Bayes (Seq-SVB) *)
  val sequential_svb : 
    model:GarchModel.t -> 
    init_var_dist:VariationalDist.t -> 
    opt_params:SVI.optimization_params -> 
    new_data:Tensor.t -> 
    VariationalDist.t * Tensor.t list * int
  
  (* Function for sequential updating with chunks of data *)
  val sequential_update : 
    model:GarchModel.t -> 
    init_var_dist:VariationalDist.t -> 
    opt_params:SVI.optimization_params -> 
    data_chunks:Tensor.t list -> 
    method_type:[`UVB | `Seq_SVB] -> 
    VariationalDist.t * Tensor.t list * float list
end