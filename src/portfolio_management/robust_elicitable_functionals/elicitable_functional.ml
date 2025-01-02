open Torch

type t = {
  name : string;
  scoring_function : Scoring_function.t;
  dimension : int;
}

let create name scoring_function dimension =
  { name; scoring_function; dimension }

let evaluate func distribution =
  let n = Tensor.shape distribution |> List.hd in
  let z = Tensor.linspace ~start:0. ~end_:1. (n * func.dimension) in
  let z = Tensor.reshape z [n; func.dimension] in
  let scores = func.scoring_function z distribution in
  let min_score, min_indices = Tensor.min2d scores in
  Tensor.select z 1 min_indices