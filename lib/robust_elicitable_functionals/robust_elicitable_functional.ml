open Torch
open Elicitable_functional

type t = {
  base_functional : Elicitable_functional.t;
  epsilon : float;
}

let create base_functional epsilon =
  { base_functional; epsilon }

let evaluate ref distribution =
  let n = Tensor.shape distribution |> List.hd in
  let z = Tensor.linspace ~start:0. ~end_:1. (n * ref.base_functional.dimension) in
  let z = Tensor.reshape z [n; ref.base_functional.dimension] in
  
  let optimize_eta z =
    let rec binary_search low high =
      if high -. low < 1e-6 then
        (low +. high) /. 2.
      else
        let mid = (low +. high) /. 2. in
        let scores = ref.base_functional.scoring_function z distribution in
        let q = Tensor.softmax (Tensor.mul scores (Tensor.of_float mid)) in
        if Kl_divergence.constraint_binding q distribution ref.epsilon then
          mid
        else if Kl_divergence.calculate q distribution > ref.epsilon then
          binary_search low mid
        else
          binary_search mid high
    in
    binary_search 0. 100.
  in

  let eta_star = optimize_eta z in
  let scores = ref.base_functional.scoring_function z distribution in
  let q_star = Tensor.softmax (Tensor.mul scores (Tensor.of_float eta_star)) in
  let weighted_scores = Tensor.mul q_star scores in
  let min_score, min_indices = Tensor.min2d weighted_scores in
  Tensor.select z 1 min_indices