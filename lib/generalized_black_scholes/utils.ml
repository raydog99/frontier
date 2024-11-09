open Torch

module Grid = struct
  type t = {
    lower: float;
    upper: float;
    n: int;
    delta: float;
    points: float array;
  }

  let make lower upper n =
    let delta = (upper -. lower) /. float_of_int n in
    let points = Array.init (n+1) (fun i -> lower +. delta *. float_of_int i) in
    { lower; upper; n; delta; points }

  let points t = t.points
  let delta t = t.delta
  let size t = t.n + 1

  let interior_points t =
    Array.sub t.points 1 (t.n - 1)

  let is_boundary t i =
    i = 0 || i = t.n
end

module Measure = struct
  type t = {
    mu1: float;
    mu2: float;
    lower: float;
    upper: float;
    max_depth: int;
  }

  let make ?(max_depth=15) lower upper mu1 =
    if mu1 < 0.0 || mu1 > 1.0 then
      failwith "mu1 must be between 0 and 1";
    { mu1; mu2 = 1.0 -. mu1; lower; upper; max_depth }

  let total_mass t = t.upper -. t.lower

  let f1 x = 0.5 *. x
  let f2 t x = 0.5 *. (x +. t.upper)

  (* Word generation for self-similar construction *)
  let rec generate_words depth =
    if depth = 0 then [[]]
    else
      let prev = generate_words (depth-1) in
      List.concat_map (fun w -> [1 :: w; 2 :: w]) prev

  let integrate f t =
    let words = generate_words t.max_depth in
    let interval_size = (t.upper -. t.lower) /. float_of_int (1 lsl t.max_depth) in
    
    let integrate_interval word =
      let x = List.fold_left (fun acc w ->
        if w = 1 then f1 acc else f2 t acc) t.lower word in
      let weight = List.fold_left (fun acc w ->
        acc *. (if w = 1 then t.mu1 else t.mu2)) 1.0 word in
      f x *. weight *. interval_size
    in
    
    List.fold_left (fun acc word -> 
      acc +. integrate_interval word) 0.0 words

  let integrate_psi t m k =
    let rec helper depth k acc =
      if depth = 0 then acc
      else
        let k' = k / 2 in
        let s_vm = k / 2 in
        let s_w = (k + 1) / 2 in
        acc *. (t.mu1 ** float_of_int s_vm *. 
               t.mu2 ** float_of_int (depth + 1 - s_vm) +.
               t.mu1 ** float_of_int (s_w + 1) *. 
               t.mu2 ** float_of_int (depth - s_w))
    in
    helper m k 1.0
end