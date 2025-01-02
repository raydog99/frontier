open Torch

module type Gaussian = sig
  type t = {
    mean: Tensor.t;
    cov: Tensor.t;
  }
  
  val create : Tensor.t -> Tensor.t -> t
  val sample : t -> int -> Tensor.t
  val log_density : t -> Tensor.t -> Tensor.t
  val cholesky : Tensor.t -> Tensor.t
end


val safe_cholesky : Tensor.t -> Tensor.t
val matrix_sqrt : Tensor.t -> Tensor.t
val matrix_invsqrt : Tensor.t -> Tensor.t
val extract_block : Tensor.t -> int -> int -> int -> Tensor.t
val set_block : Tensor.t -> Tensor.t -> int -> int -> unit
val is_positive_definite : Tensor.t -> Tensor.t
val kron : Tensor.t -> Tensor.t -> Tensor.t

val matrix_log : Tensor.t -> Tensor.t
val matrix_exp : Tensor.t -> Tensor.t
val matrix_power_pd : Tensor.t -> float -> Tensor.t
val solve_sylvester : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val matrix_gmean : Tensor.t -> Tensor.t -> Tensor.t
val project_psd : Tensor.t -> Tensor.t

val frechet_derivative : (Tensor.t -> Tensor.t) -> 
                        Tensor.t -> Tensor.t -> Tensor.t
val matrix_gradient : (Tensor.t -> Tensor.t) -> 
                     Tensor.t -> Tensor.t
val matrix_hessian : (Tensor.t -> Tensor.t) -> 
                    Tensor.t -> Tensor.t

val kl_divergence : Gaussian.t -> Gaussian.t -> Tensor.t
val quadratic_cost : Tensor.t -> Tensor.t -> Tensor.t
val entropic_wasserstein_2 : ?lambda:float -> Gaussian.t -> Gaussian.t -> Tensor.t

val f_lambda : float -> Tensor.t -> Tensor.t

val compute_trace_terms : Tensor.t -> Tensor.t -> 
                        Tensor.t -> Tensor.t -> Tensor.t
val compute_det_term : float -> Tensor.t -> Tensor.t
val adapted_wasserstein_2 : ?lambda:float -> Gaussian.t -> Gaussian.t -> Tensor.t

val construct_optimal_coupling : Gaussian.t -> Gaussian.t -> float -> Gaussian.t

module type DPControl = sig
  type value_function = {
    value: Tensor.t;
    gradient: Tensor.t;
  }

  type policy = {
    coupling: Gaussian.t;
    next_state: Tensor.t -> Tensor.t * Tensor.t;
  }

  module ValueIteration : sig
    val backward : Gaussian.t -> Gaussian.t -> float -> 
                  value_function list * policy list
  end
end