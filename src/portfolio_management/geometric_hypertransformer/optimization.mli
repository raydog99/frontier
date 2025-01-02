open Torch

type optimizer_state = {
  parameters: Tensor.t;
  velocity: Tensor.t;
  momentum: float;
  learning_rate: float;
}

val riemannian_sgd:
  InformationGeometry.statistical_manifold ->
  optimizer_state ->
  Tensor.t ->
  optimizer_state

val riemannian_adam:
  InformationGeometry.statistical_manifold ->
  optimizer_state ->
  Tensor.t ->
  optimizer_state