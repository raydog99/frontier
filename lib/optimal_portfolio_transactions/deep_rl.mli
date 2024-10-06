open Torch
open Optimal_execution

type config = {
  state_dim: int;
  action_dim: int;
  hidden_dim: int;
  num_layers: int;
  learning_rate: float;
  discount_factor: float;
  replay_buffer_size: int;
  batch_size: int;
  target_update_freq: int;
}

type state

val create_config :
  state_dim:int ->
  action_dim:int ->
  hidden_dim:int ->
  num_layers:int ->
  learning_rate:float ->
  discount_factor:float ->
  replay_buffer_size:int ->
  batch_size:int ->
  target_update_freq:int ->
  config

val init : config -> state
val select_action : state -> Tensor.t -> Tensor.t
val update : state -> state
val train : Optimal_execution.t -> int -> state -> state