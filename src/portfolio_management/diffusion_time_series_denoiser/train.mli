open Torch
open Score_network
open Sde

val train : Score_network.t -> SDE -> Dataset.t -> int -> float -> unit