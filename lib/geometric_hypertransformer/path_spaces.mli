open Torch

type 'a path = {
  values: 'a array;
  times: float array;
  filtration: Tensor.t array;
  regularity: [`Holder of float * float 
             | `Exponential of float * float array
             | `Weighted of (int -> float) * float];
}

val verify_regularity: 'a path -> [`Regular | `Irregular of float list]
val decompose: 'a path -> int -> 'a path * 'a path
val path_distance: 'a path -> 'a path -> float