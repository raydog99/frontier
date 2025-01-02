open Torch
open Types

(** Verify Matrix Bernstein inequality *)
val bernstein : 
  Tensor.t ->
  float ->
  float ->
  float ->
  float ->
  bool

(** Verify tail bounds *)
val tail_bounds :
  Tensor.t ->
  float ->
  float list ->
  bool list

(** Verify covariance concentration *)
val covariance_concentration :
  Tensor.t ->
  float ->
  float ->
  float ->
  bool

(** Verify all bounds simultaneously *)
val all_bounds :
  Tensor.t ->
  markov_chain ->
  float ->
  float ->
  float ->
  Gpu_compute.device_config ->
  bool * string list