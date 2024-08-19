open Torch

type t = {
  a: Tensor.t;
  b: Tensor.t;
  a_bias: Tensor.t;
  b_bias: Tensor.t;
  reference_paths: Tensor.t;
}

let create n_ref n_sim d_y ~device =
  let a = Tensor.randn [n_sim; n_sim] ~device in
  let b = Tensor.randn [n_sim; n_ref] ~device in
  let a_bias = Tensor.randn [n_sim] ~device in
  let b_bias = Tensor.randn [n_sim] ~device in
  let reference_paths = Tensor.randn [n_ref; d_y] ~device in
  { a; b; a_bias; b_bias; reference_paths }

let forward t y =
  let distances = Tensor.(y - t.reference_paths)
                  |> Tensor.norm ~p:(Scalar.i 2) ~dim:[1] ~keepdim:false in
  let x = Tensor.(matmul t.b distances + t.b_bias) in
  let x = Tensor.relu x in
  let x = Tensor.(matmul t.a x + t.a_bias) in
  Tensor.softmax x ~dim:[0] ~dtype:(T Float)

let parameters t =
  [t.a; t.b; t.a_bias; t.b_bias]