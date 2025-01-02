open Torch

type parameter = {
  mutable value: float;
  mutable gradient: float;
}

type basis_function = {
  f: float array -> float;
  parameters: parameter array;
}

type dictionary = basis_function array

type optimization_config = {
  learning_rate: float;
  max_iterations: int;
  batch_size: int;
  tolerance: float;
  validation_fraction: float;
}

module KoopmanOperator : sig
  type flow_map = {
    forward: Torch.Tensor.t -> float -> Torch.Tensor.t;
    drift: Torch.Tensor.t -> Torch.Tensor.t;
    diffusion: Torch.Tensor.t -> Torch.Tensor.t;
  }

  val create_sde_system : 
    drift:(Torch.Tensor.t -> Torch.Tensor.t) ->
    diffusion:(Torch.Tensor.t -> Torch.Tensor.t) ->
    dt:float ->
    flow_map

  val evolve_observable : 
    flow_map -> 
    (Torch.Tensor.t -> Torch.Tensor.t) -> 
    Torch.Tensor.t -> 
    float -> 
    Torch.Tensor.t
end

module EDMD : sig
  type t

  val create : dictionary -> t
  val compute_k_matrix : t -> Torch.Tensor.t -> Torch.Tensor.t -> Torch.Tensor.t
  val compute_eigendecomposition : t -> Torch.Tensor.t * Torch.Tensor.t
  val reconstruction_error : t -> Torch.Tensor.t -> Torch.Tensor.t -> float
end

module SINDy : sig
  type t

  val create : dictionary -> float -> t
  val identify_system : t -> Torch.Tensor.t -> float -> Torch.Tensor.t
  val reconstruction_error : t -> Torch.Tensor.t -> float -> float
end

module Optimization : sig
  type optimization_result = {
    parameters: Torch.Tensor.t;
    loss_history: float array;
    iterations: int;
    converged: bool;
    training_time: float;
  }

  module EDMD : sig
    val train : 
      EDMD.t -> 
      optimization_config -> 
      Torch.Tensor.t -> 
      Torch.Tensor.t -> 
      optimization_result
  end

  module SINDy : sig
    val train : 
      SINDy.t -> 
      optimization_config -> 
      Torch.Tensor.t -> 
      float -> 
      optimization_result
  end

  module PDEFind : sig
    type grid_info = {
      dx: float;
      dt: float;
      x_points: int;
      t_points: int;
    }

    val train : 
      dictionary -> 
      optimization_config -> 
      grid_info ->
      Torch.Tensor.t -> 
      optimization_result
  end
end

module ModelSelection : sig
  type validation_result = {
    training_loss: float;
    validation_loss: float;
    training_time: float;
    iterations: int;
    sparsity_level: float;
    selected_terms: string array;
  }

  val cross_validate : 
    < train : 'a -> 'b
    ; evaluate : 'a -> 'c -> float
    ; .. > ->
    'b -> int -> validation_result array

  val compare_models : 
    (string * (< train : 'a -> 'b
             ; evaluate : 'a -> 'c -> float
             ; .. > as 'd)) array ->
    'b -> (string * float * validation_result array) array
end

module type DynamicalSystem = sig
  type t
  type config
  type data
  type result

  val create : config -> t
  val train : t -> data -> result
  val predict : t -> Torch.Tensor.t -> Torch.Tensor.t
  val get_parameters : t -> parameter array
  val save_model : t -> string -> unit
  val load_model : string -> t
end

module EDMDLearner : DynamicalSystem
module SINDyLearner : DynamicalSystem
module PDELearner : DynamicalSystem

val create_learner : 
  [< `EDMD | `SINDy | `PDEFind ] -> 
  optimization_config -> 
  (module DynamicalSystem)

val train_model : 
  [< `EDMD | `SINDy | `PDEFind ] ->
  Torch.Tensor.t * Torch.Tensor.t ->
  optimization_config ->
  'a * optimization_result * validation_result

val save_model : < get_parameters : parameter array; .. > -> string -> unit
val load_model : 
  [< `EDMD | `SINDy | `PDEFind ] -> 
  string -> 
  (module DynamicalSystem)