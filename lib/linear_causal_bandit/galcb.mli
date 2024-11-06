open Torch

type observation = {
  values: Tensor.t;
  intervention: Graph.NodeSet.t;
  timestamp: int;
  reward: float;
}

type config = {
  alpha: float;
  min_exploration_rounds: int;
  phases: int;
  epsilon: float;
}

type state

val default_config : config

val create : Graph.t -> Linear_sem.t -> float -> int -> ?config:config -> unit -> state

module StructureLearning : sig
  val learn : state -> Graph.NodeSet.t Graph.NodeMap.t
  val calculate_exploration_rounds : int -> float -> float -> int * int
end

module InterventionDesign : sig
  val select_action : state -> Graph.NodeSet.t
  val update : state -> observation -> state
  val calculate_reward : Tensor.t -> Graph.NodeSet.t -> float
end

val run : state -> int -> state