open Torch
open Similiary_score
open Positional_encoding

type t = {
  similarity_score: Similiarity_score.t;
  positional_encoding: Positional_encoding.t;
  c: Tensor.t;
}

let create n_ref n_sim d_y n_pos n_time ~device =
  let similarity_score = Similiarity_score.create n_ref n_sim d_y ~device in
  let positional_encoding = Positional_encoding.create n_pos d_y n_time ~device in
  let c = Tensor.randn [n_sim * d_y] ~device in
  { similarity_score; positional_encoding; c }

let forward t y =
  let sim_scores = Similiarity_score.forward t.similarity_score y in
  let pos_encoding = Positional_encoding.forward t.positional_encoding y in
  let features = Tensor.(sim_scores * pos_encoding) in
  Tensor.matmul t.c (Tensor.view features ~size:[-1; 1])

let parameters t =
  Similiarity_score.parameters t.similarity_score @
  Positional_encoding.parameters t.positional_encoding @
  [t.c]