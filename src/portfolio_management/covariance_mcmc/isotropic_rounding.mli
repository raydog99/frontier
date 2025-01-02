open Torch
open Types

type rounding_config = {
  radius: float;
  step_size: float;
  n_iterations: int;
  batch_size: int;
  tolerance: float;
}

type rounding_state = {
  transform: Tensor.t;
  radius: float;
  samples: Tensor.t;
  iteration: int;
}

val check_isotropic : Tensor.t -> float -> bool

val isotropize : 
  (Tensor.t -> bool) ->  (* Membership oracle *)
  int ->                 (* Dimension *)
  Tensor.t ->           (* Initial transform *)
  rounding_config ->
  distribution

(** Memory-efficient version for large dimensions *)
val isotropize_memory_efficient :
  (Tensor.t -> bool) ->
  int ->
  Tensor.t ->
  Batch_processing.batch_config ->
  distribution * float array