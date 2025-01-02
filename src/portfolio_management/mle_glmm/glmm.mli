open Torch
open Types

type t = {
  spec: Types.model_spec;
  x: Tensor.t;         (** Fixed effects design matrix *)
  z: Tensor.t;         (** Random effects design matrix *)
  y: Tensor.t;         (** Response vector *)
  beta: Tensor.t;      (** Fixed effects parameters *)
  omega: Tensor.t;     (** Variance components *)
}

val create : Types.model_spec -> Tensor.t -> Tensor.t -> 
            Tensor.t -> t
(** [create spec x z y] creates a new GLMM instance *)

val linear_predictor : t -> Tensor.t -> Tensor.t
(** [linear_predictor t gamma] computes Xβ + Zγ *)

val compute_working_response : t -> Tensor.t -> 
                             Tensor.t * Tensor.t
(** [compute_working_response t gamma] returns (y_tilde, w) *)

val compute_r : t -> Tensor.t -> Tensor.t -> Tensor.t
(** [compute_r t w delta] computes R matrix *)

val compute_psi : t -> Tensor.t -> Tensor.t -> 
                 Tensor.t -> Tensor.t
(** [compute_psi t alpha delta gamma] computes objective function ψ *)

val gradient_alpha : t -> Tensor.t -> Tensor.t -> 
                    Tensor.t -> Tensor.t -> Tensor.t
(** [gradient_alpha t alpha delta gamma y_tilde] computes gradient wrt α *)

val gradient_delta : t -> Tensor.t -> Tensor.t -> 
                    Tensor.t -> Tensor.t -> Tensor.t
(** [gradient_delta t alpha delta gamma y_tilde] computes gradient wrt δ *)

val update : t -> t
(** [update t] performs one update iteration *)

val fit : ?max_iter:int -> ?tol:float -> t -> t
(** [fit ?max_iter ?tol t] fits the model *)