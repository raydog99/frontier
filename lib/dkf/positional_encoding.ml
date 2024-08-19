open Torch

type t = {
  v: Tensor.t;
  u: Tensor.t;
  query_times: Tensor.t;
}

let create n_pos d_y n_time ~device =
  let v = Tensor.randn [n_pos; d_y] ~device in
  let u = Tensor.randn [n_pos; n_time] ~device in
  let query_times = Tensor.linspace ~start:0. ~end_:1. n_time ~device in
  { v; u; query_times }

let forward t y =
  let y_sampled = Tensor.index_select y ~dim:0 ~index:t.query_times in
  Tensor.(matmul t.u y_sampled + t.v)

let parameters t =
  [t.v; t.u]