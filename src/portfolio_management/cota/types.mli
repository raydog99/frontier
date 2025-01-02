open Torch

type scm = {
  variables: string array;
  domains: int array;
  graph: (int * int) list;
}

type intervention = {
  variables: int array;
  values: float array;
  func: Tensor.t -> Tensor.t;
}

type transport_plan = {
  plan: Tensor.t;
  source_dist: Tensor.t;
  target_dist: Tensor.t;
}

type abstraction_map = {
  tau: Tensor.t -> Tensor.t;
  omega: intervention -> intervention;
}

type chain = intervention list

type comparability = 
  | Comparable of intervention list
  | NotComparable

val intervention_leq : intervention -> intervention -> bool
val is_compatible : Tensor.t -> intervention -> bool