open Torch
open Types

type optimizer_state = {
  params: model_params;
  grad: Tensor.t;
  momentum: Tensor.t option;
  iteration: int;
}

val adam : float -> float * float -> float -> 
          optimizer_state -> optimizer_state

val newton_raphson : data -> model_params ->
                    int -> float -> model_params