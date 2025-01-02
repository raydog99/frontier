open Torch
open Types

type t = {
  input_layer : Tensor.t;
  hidden_layer : Tensor.t;
  output_layer : Tensor.t;
  b_spline : Tensor.t;
}

let create params =
  if params.input_dim <= 0 || params.hidden_dim <= 0 || params.output_dim <= 0 then
    raise (Invalid_parameter "KAN dimensions must be positive");
  let input_layer = Tensor.randn [params.input_dim; params.hidden_dim] ~requires_grad:true in
  let hidden_layer = Tensor.randn [params.hidden_dim; params.hidden_dim] ~requires_grad:true in
  let output_layer = Tensor.randn [params.hidden_dim; params.output_dim] ~requires_grad:true in
  let b_spline = Tensor.randn [params.hidden_dim; 4] ~requires_grad:true in
  { input_layer; hidden_layer; output_layer; b_spline }

let b_spline_basis x =
  let open Tensor in
  let t = x * scalar 3. in
  let t2 = t * t in
  let t3 = t2 * t in
  stack [
    scalar 1. - t + t2 * scalar 0.5 - t3 * scalar (1. /. 6.);
    t2 * scalar 0.5 - t3 * scalar 0.5;
    t3 * scalar (1. /. 6.);
    t3 * scalar (1. /. 6.)
  ] ~dim:1

let forward t x =
  let open Tensor in
  let h1 = matmul x t.input_layer in
  let h1_spline = map (fun xi -> matmul (b_spline_basis xi) t.b_spline) h1 in
  let h2 = matmul h1_spline t.hidden_layer in
  let h2_spline = map (fun xi -> matmul (b_spline_basis xi) t.b_spline) h2 in
  matmul h2_spline t.output_layer

let parameters t =
  [t.input_layer; t.hidden_layer; t.output_layer; t.b_spline]