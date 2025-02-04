open Torch

type direction = X_to_Y | Y_to_X

type validation_error = 
  | InvalidProbabilities
  | InvalidDimensions
  | NumericalInstability
  | InvalidFunction

type validation_result = 
  | Valid 
  | Invalid of validation_error

type causal_pair = {
  x_dist: Tensor.t;
  y_dist: Tensor.t;
  conditional: Tensor.t;  (** Y|X or X|Y *)
  nx: int;
  ny: int;
}

type exogenous_variable = {
  distribution: Tensor.t;
  size: int;
  entropy: float;
}

module BlockPartition : sig
  type t = {
    matrix: Tensor.t;
    num_blocks: int;
    block_size: int;
  }

  val validate_partition : Tensor.t -> int -> validation_result
  val from_conditional : Tensor.t -> (t, validation_error) result
  val to_conditional_probability : t -> Tensor.t
end

module Entropy : sig
  val eps : float
  val h0_entropy : Tensor.t -> float
  val h1_entropy : Tensor.t -> float
  val conditional_entropy : Tensor.t -> float
  val mutual_information : Tensor.t -> float
  val min_required_exogenous_entropy : Tensor.t -> float
end

val function_constraints : Tensor.t -> validation_result
val get_inverse_maps : Tensor.t -> float list array array
val greedy_minimize : Tensor.t array -> exogenous_variable
val find_minimum_h0_exogenous : causal_pair -> exogenous_variable
val infer_direction : causal_pair -> (direction, validation_error) result
val entropy_gap : causal_pair -> float
val create_from_data : Tensor.t -> Tensor.t -> causal_pair

val infer_causality : Tensor.t -> Tensor.t -> (direction * float, validation_error) result