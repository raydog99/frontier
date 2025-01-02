open Torch
open Types

type proximal_config = {
  step_size: float;
  n_steps: int;
  beta: float;      (* Smoothness parameter *)
  epsilon: float;   (* Target accuracy *)
}

val proximal_step :
  (Tensor.t -> Tensor.t) ->  (* Gradient function *)
  Tensor.t ->               (* Current point *)
  Tensor.t ->               (* Target point *)
  float ->                  (* Step size *)
  Tensor.t

val estimate_covariance_with_guarantee :
  (Tensor.t -> Tensor.t) ->  (* Gradient function *)
  int ->                    (* Dimension *)
  float ->                  (* Beta (smoothness) *)
  float ->                  (* Epsilon (accuracy) *)
  distribution * bool       (* Result and guarantee verification *)

val estimate_high_dim :
  (Tensor.t -> Tensor.t) ->
  int ->
  proximal_config ->
  Gpu_compute.device_config ->
  distribution * Convergence_diagnostics.convergence_stats