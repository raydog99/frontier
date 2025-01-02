open Torch
open Types
open Utils

(* Core convolution operation *)
let convolve p_z_x distance threshold y z =
  let kernel = match threshold with
    | Exponential alpha ->
        let d = match distance with
          | Quadratic -> 
              Tensor.(y - z |> sqr |> sum)
          | KLDivergence -> 
              (* Create empirical distributions for y and z *)
              let p_y = normalize (Tensor.ones_like y) in
              let p_z = normalize (Tensor.ones_like z) in
              kl_divergence p_y p_z
        in
        Tensor.(exp (Scalar.neg alpha * d))
  in
  (* Compute convolution and normalize *)
  Tensor.(kernel * p_z_x) |> normalize

(* Convolutional transition probability *)
let conv_transition_prob transition_fn x_prev x_t params distance threshold =
  (* Get nominal transition probability *)
  let p_xbar = transition_fn x_prev x_t params in
  
  (* Apply convolution operation *)
  convolve p_xbar distance threshold x_t x_t
  
(* Convolutional measurement probability *)
let conv_measurement_prob measurement_fn x_t y_t params distance threshold =
  (* Get nominal measurement probability *)
  let p_ybar = measurement_fn x_t y_t params in
  
  (* Apply convolution operation *)
  convolve p_ybar distance threshold y_t y_t