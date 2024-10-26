open Torch

type state = {
  beta: Tensor.t;
  omega: Tensor.t;
  gamma: Tensor.t;
  eta: Tensor.t;
  y_tilde: Tensor.t;
  w: Tensor.t;
  psi: Tensor.t;
  converged: bool;
}

val initialize_state : Glmm.t -> state
(** [initialize_state model] creates initial state for the algorithm *)

val update_working_values : Glmm.t -> state -> Tensor.t -> state
(** [update_working_values model state y] updates working response and weights *)

val newton_raphson_update : Glmm.t -> state -> state
(** [newton_raphson_update model state] performs Newton-Raphson update *)

val check_convergence : state -> state -> float -> state
(** [check_convergence state prev_state tol] checks convergence *)

val iterate : Glmm.t -> state -> state
(** [iterate model state] performs one iteration *)

val run : ?max_iter:int -> ?tol:float -> Glmm.t -> state
(** [run ?max_iter ?tol model] runs the full algorithm *)