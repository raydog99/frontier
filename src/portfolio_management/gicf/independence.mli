open Torch
open Types

type test_result = {
  statistic: float;
  p_value: float;
  test_method: independence_test_method;
  sample_size: int;
  degrees_freedom: int;
  confidence_interval: float * float;
}

val test_marginal_independence : Tensor.t -> int -> int -> independence_test_method -> test_result

val learn_independence_graph : ?alpha:float -> Tensor.t -> graph

val compute_partial_correlation : Tensor.t -> int -> int -> Tensor.t -> float