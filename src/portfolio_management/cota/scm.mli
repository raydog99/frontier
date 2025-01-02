open Torch

type noise_dist = 
  | Normal of float * float
  | Uniform of float * float
  | Custom of (unit -> float)

type structural_fn = {
  fn: Tensor.t -> Tensor.t -> Tensor.t;
  noise: noise_dist;
}

type t = {
  variables: string array;
  domains: int array;
  graph: (int * int) list;
  functions: structural_fn array;
}

val sample_noise : noise_dist -> Tensor.t
val topological_sort : t -> int list
val sample : t -> int -> Tensor.t
val apply_intervention : t -> Types.intervention -> t