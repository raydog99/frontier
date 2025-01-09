open Torch

type space = {
  dim: int;
  metric: Tensor.t -> Tensor.t -> float;
  is_compact: bool;
}

type measure = {
  weights: Tensor.t;
  points: Tensor.t;
  space: space;
}

type cost = {
  source_space: space;
  target_space: space;
  fn: Tensor.t -> Tensor.t -> Tensor.t;
  is_continuous: bool;
}

type transport_plan = {
  plan: Tensor.t;
  source: Tensor.t;
  target: Tensor.t;
}

type coupling = {
  measure: Tensor.t;
  marginals: Tensor.t list;
  support: Tensor.t list;
}

val logsumexp : Tensor.t -> dim:int list -> Tensor.t
val matrix_sqrt : Tensor.t -> Tensor.t
val is_positive_definite : Tensor.t -> bool
val nearest_pd : Tensor.t -> Tensor.t

val compute_cost_matrix : 
  source:Tensor.t -> 
  target:Tensor.t -> 
  cost:(Tensor.t -> Tensor.t -> Tensor.t) ->
  Tensor.t

val verify_optimality :
  Tensor.t ->
  Tensor.t ->
  Tensor.t ->
  Tensor.t ->
  bool

val compute_b :
  costs:cost list ->
  points:Tensor.t list ->
  weights:Tensor.t list ->
  Tensor.t

val solve_sinkhorn :
  cost_matrix:Tensor.t ->
  source_weights:Tensor.t ->
  target_weights:Tensor.t ->
  epsilon:float ->
  Tensor.t

val solve_exact :
  cost_matrix:Tensor.t ->
  source_weights:Tensor.t ->
  target_weights:Tensor.t ->
  Tensor.t

module Fixed_point : sig
  val compute_next :
    measure ->
    measure list ->
    cost list ->
    measure

  val iterate :
    init:measure ->
    targets:measure list ->
    costs:cost list ->
    max_iter:int ->
    measure

  val check_convergence :
    measure list ->
    bool
end

val iterate_g :
  init:measure ->
  targets:measure list ->
  costs:cost list ->
  max_iter:int ->
  measure

val iterate_h :
  init:measure ->
  targets:measure list ->
  costs:cost list ->
  max_iter:int ->
  measure

module GMM : sig
  type gaussian = {
    mean: Tensor.t;
    covariance: Tensor.t;
  }

  type gmm = {
    components: gaussian list;
    weights: Tensor.t;
  }

  val bures_wasserstein :
    gaussian ->
    gaussian ->
    Tensor.t

  val fixed_point_iteration :
    Tensor.t ->
    gaussian list ->
    float list ->
    Tensor.t

  val compute_barycentre :
    gmm list ->
    float list ->
    int ->
    gaussian
end