open Torch

type chain_state = {
  samples: Type.posterior_sample list;
  rng_state: Random.State.t;
  id: int;
}

val init_chains : int -> chain_state array
(** [init_chains n_chains] initializes parallel chains *)

val run_chain : 
  Tensor.t -> Tensor.t -> Type.projection_config -> chain_state -> chain_state
(** [run_chain x y config state] runs single chain with own RNG state *)

val run_parallel_chains : 
  Tensor.t -> Tensor.t -> Type.projection_config -> int -> chain_state array
(** [run_parallel_chains x y config n_chains] runs chains in parallel using Domain *)