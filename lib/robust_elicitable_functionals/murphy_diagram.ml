open Torch
open Elicitable_functional

type t = {
  functional: Elicitable_functional.t;
  b_values: float array;
  distribution: Tensor.t;
}

let create functional b_values distribution =
  { functional; b_values; distribution }

let evaluate murphy_diagram =
  let n = Array.length murphy_diagram.b_values in
  let results = Array.make n 0. in
  for i = 0 to n - 1 do
    let b = murphy_diagram.b_values.(i) in
    let b_func = 
      match murphy_diagram.functional.name with
      | name when String.sub name 0 3 = "VaR" ->
          let alpha = float_of_string (String.sub name 4 (String.length name - 4)) in
          B_homogeneous.var b alpha
      | name when String.sub name 0 2 = "ES" ->
          let alpha = float_of_string (String.sub name 3 (String.length name - 3)) in
          B_homogeneous.es b alpha
      | _ -> B_homogeneous.mean b
    in
    let result = Elicitable_functional.evaluate b_func murphy_diagram.distribution in
    results.(i) <- Tensor.to_float0_exn result
  done;
  results