val compute_acf : float array -> int -> float array
(** [compute_acf series max_lag] computes autocorrelation function *)

val compute_ess : float array -> float
(** [compute_ess series] computes effective sample size *)

val compute_psrf : float array array -> float
(** [compute_psrf chains] computes potential scale reduction factor (Gelman-Rubin) *)

val assess_convergence : 
  Type.posterior_sample list -> Type.posterior_sample array array -> 
  Type.convergence_diagnostics array
(** [assess_convergence samples chains] assesses convergence for all parameters *)