open Torch

module Linear : sig
  type belief_structure = {
    mean_x: Tensor.t;
    mean_d: Tensor.t;
    var_x: Tensor.t;
    var_d: Tensor.t;
    cov_xd: Tensor.t;
    inner_product_space: LinearAlgebra.InnerProduct.t;
  }

  val create_belief_structure : 
    mean_x:Tensor.t -> 
    mean_d:Tensor.t -> 
    var_x:Tensor.t -> 
    var_d:Tensor.t -> 
    cov_xd:Tensor.t -> 
    belief_structure

  val adjusted_expectation : belief_structure -> Tensor.t -> Tensor.t
  val adjusted_variance : belief_structure -> Tensor.t

  module Sequential : sig
    type update_sequence = {
      updates: Tensor.t list;
      initial_belief: belief_structure;
    }

    val create_sequence : 
      initial_belief:belief_structure -> 
      updates:Tensor.t list -> 
      update_sequence

    val apply_sequence : 
      update_sequence -> 
      (Tensor.t * Tensor.t)
  end

  module Diagnostics : sig
    type diagnostic_result = {
      correlation_valid: bool;
      variance_positive: bool;
      dimension_valid: bool;
      condition_numbers: float list;
    }

    val check_belief_structure : belief_structure -> diagnostic_result
    val suggest_improvements : diagnostic_result -> string list
  end
end

module Generalized : sig
  module Spaces : sig
    type t = {
      compute: Tensor.t -> Tensor.t -> Tensor.t;
      gradient: Tensor.t -> Tensor.t -> Tensor.t;
      is_symmetric: bool;
      satisfies_triangle: bool;
    }

    val kl_divergence : unit -> t
    val alpha_divergence : float -> t
    val total_variation : unit -> t
  end

  module Spaces : sig
    type t = {
      project: Tensor.t -> Tensor.t;
      is_member: Tensor.t -> bool;
      is_convex: bool;
      dimension: int;
    }

    val probability_simplex : int -> t
    val bounded : int -> float -> float -> t
  end

  type inference_params = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
  }

  type t = {
    solution_space: Spaces.t;
    divergence: Spaces.t;
    polish_space: Spaces.Polish.t;
  }

  val create : 
    dim:int -> 
    divergence:Spaces.t -> 
    solution_space:Spaces.t -> 
    t

  module Optimization : sig
    type optimization_state = {
      point: Tensor.t;
      value: float;
      gradient_norm: float;
      iteration: int;
    }

    val adam : 
      learning_rate:float ->
      beta1:float ->
      beta2:float ->
      epsilon:float ->
      Tensor.t ->
      Tensor.t

    val gradient_descent_momentum : 
      learning_rate:float ->
      momentum:float ->
      Tensor.t ->
      Tensor.t
  end

  val infer : 
    system:t ->
    prior:Tensor.t ->
    data:Tensor.t ->
    params:inference_params ->
    Tensor.t

  val infer_with_loss : 
    system:t ->
    prior:Tensor.t ->
    data:Tensor.t ->
    loss_fn:(Tensor.t -> Tensor.t -> Tensor.t) ->
    params:inference_params ->
    Tensor.t
end

module ConjugateFamilies : sig
  type distribution_family = 
    | Gaussian
    | Gamma
    | Beta
    | Poisson
    | Binomial

  type sufficient_statistics = {
    compute: Tensor.t -> Tensor.t;
    dim: int;
  }

  type natural_parameters = {
    to_natural: Tensor.t -> Tensor.t;
    from_natural: Tensor.t -> Tensor.t;
    valid_space: Tensor.t -> bool;
  }

  val gaussian_parameters : unit -> natural_parameters
  val gaussian_statistics : sufficient_statistics
  val verify_posterior_linearity : 
    distribution_family -> 
    Tensor.t -> 
    Tensor.t -> 
    bool
end