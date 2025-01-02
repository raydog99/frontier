open Torch

type config = {
  lambda: float;
  gamma: float;
  epsilon: float;
  max_iter: int;
  lr: float;
  tol: float;
}

val chain_objective : 
  Types.transport_plan list -> Types.chain -> 
  Tensor.t -> config -> Tensor.t
val optimize_chain : 
  Types.chain -> Tensor.t -> Tensor.t ->
  (Types.intervention -> Types.intervention) -> config -> Types.transport_plan list
val create_abstraction_map :
  Types.transport_plan list list -> 
  (Types.intervention -> Types.intervention) -> Types.abstraction_map
val optimize : 
  Scm.t -> Scm.t -> Types.intervention list -> config -> Types.abstraction_map