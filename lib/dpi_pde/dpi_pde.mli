open Torch

module PDE : sig
  type t = {
    dim : int;
    time_horizon : float;
    terminal_condition : Tensor.t -> Tensor.t;
    nonlinearity : Tensor.t -> Tensor.t -> Tensor.t;
  }
end

module Network : sig
  type t = {
    layers : nn;
    device : Device.t;
  }

  val create : input_dim:int -> hidden_dim:int -> num_layers:int -> device:Device.t -> t
  val forward : t -> Tensor.t -> Tensor.t
  val gradient : t -> Tensor.t -> Tensor.t
  val create_optimized : input_dim:int -> hidden_dim:int -> num_layers:int -> device:Device.t -> t
end

module SDE : sig
  type t = 
    | Brownian
    | GeometricBrownian of float
    | OrnsteinUhlenbeck of float

  val drift : t -> Tensor.t -> Tensor.t
  val diffusion : t -> Tensor.t -> Tensor.t
  val sample_path : t -> batch_size:int -> dim:int -> num_steps:int -> dt:float -> device:Device.t -> Tensor.t
end

module MC : sig
  type path = {
    times : Tensor.t;
    values : Tensor.t;
    increments : Tensor.t;
  }

  val feynman_kac_estimate : 
    pde:PDE.t -> 
    network:Network.t -> 
    x_t:Tensor.t -> 
    t:Tensor.t -> 
    dt:float -> 
    num_samples:int -> 
    device:Device.t -> 
    Tensor.t

  val gradient_finite_var :
    pde:PDE.t ->
    network:Network.t ->
    x_t:Tensor.t ->
    t:Tensor.t ->
    dt:float ->
    num_samples:int ->
    Tensor.t
end

module PicardIteration : sig
  type config = {
    dt : float;
    num_steps : int;
    batch_size : int;
    device : Device.t;
  }

  val iterate : 
    pde:PDE.t -> 
    network:Network.t -> 
    x_t:Tensor.t -> 
    t:Tensor.t -> 
    config:config -> 
    (Tensor.t * Tensor.t * Tensor.t) list

  val compute_next_iterate :
    pde:PDE.t ->
    network:Network.t ->
    paths:(Tensor.t * Tensor.t * Tensor.t) list ->
    config:config ->
    Tensor.t
end

module Training : sig
  type config = {
    batch_size : int;
    learning_rate : float;
    num_epochs : int;
    num_mc_samples : int;
    picard_iterations : int;
    dt : float;
    lambda : float;
  }

  val train : 
    pde:PDE.t ->
    network:Network.t ->
    config:config ->
    device:Device.t ->
    unit
end

module Optimization : sig
  module Parallel : sig
    type batch_config = {
      total_size : int;
      device_batch_size : int;
      num_gpus : int;
      main_device : Device.t;
    }

    val distribute_batch : 
      config:batch_config -> 
      data:Tensor.t -> 
      (Tensor.t * Device.t) list

    val parallel_forward :
      network:Network.t ->
      batches:(Tensor.t * Device.t) list ->
      Tensor.t list
  end

  module Memory : sig
    val checkpoint_forward :
      network:Network.t ->
      input:Tensor.t ->
      checkpoint_layers:nn list ->
      Tensor.t * (nn * Tensor.t) list

    val stream_batches :
      total_size:int ->
      batch_size:int ->
      f:(int -> int -> 'a) ->
      'a list
  end
end