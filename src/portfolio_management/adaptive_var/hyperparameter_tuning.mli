open Types
open Config

type hyperparameters = {
  theta: float;
  r: float;
  ca: float;
  m: int;
}

val bayesian_optimization : t -> framework -> loss_function -> float -> int -> hyperparameters * float