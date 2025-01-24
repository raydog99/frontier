open Torch

(* Domain *)
type domain =
  | WholeSpace of int  (** R^d with dimension *)
  | Torus of int       (** T^d with dimension *)

(* Potentials *)
module Potential : sig
  type t

  val create : grad:(Tensor.t -> Tensor.t) ->
               hessian:(Tensor.t -> Tensor.t) ->
               domain:domain ->
               convexity_constant:float ->
               t

  val check_uniform_convexity : t -> Tensor.t -> Tensor.t -> bool
  val convolve : t -> (Tensor.t -> Tensor.t) -> Tensor.t -> int -> Tensor.t
end

(* Distributions *)
module Distribution : sig
  type t

  val create : density:(Tensor.t -> Tensor.t) ->
               log_density:(Tensor.t -> Tensor.t) ->
               domain:domain ->
               t

  val relative_entropy : ?num_samples:int -> t -> t -> float
  val relative_fisher_info : ?num_samples:int -> t -> t -> float
end

(* McKean SDE *)
module McKeanSDE : sig
  type t

  val create : v:Potential.t -> w:Potential.t -> beta:float -> domain:domain -> t
  val drift : t -> Tensor.t -> (Tensor.t -> Tensor.t) -> Tensor.t
  val diffusion : t -> float
  val evolve : t -> Tensor.t -> (Tensor.t -> Tensor.t) -> float -> Tensor.t
  
  val solve_invariant_measure : t -> 
    initial_guess:(Tensor.t -> Tensor.t) ->
    max_iter:int ->
    tolerance:float ->
    t
end

(* Linearized Process *)
module LinearizedProcess : sig
  type t

  val create : McKeanSDE.t -> t
  val drift : t -> Tensor.t -> Tensor.t
  val evolve : t -> Tensor.t -> float -> Tensor.t
end

(* Fokker-Planck Solver *)
module FokkerPlanck : sig
  type scheme = 
    | Explicit
    | CrankNicolson
    | ADI

  type solution = {
    times: float array;
    densities: (Tensor.t -> Tensor.t) array;
    scheme: scheme;
    dt: float;
  }

  val solve : initial:(Tensor.t -> Tensor.t) ->
              t_final:float ->
              dt:float ->
              scheme:scheme ->
              solution
end

(* Convergence analysis *)
module ConvergenceAnalysis : sig
  type convergence_metric = 
    | RelativeEntropy
    | L2Distance
    | WassersteinDistance

  val compute_entropy_convergence : h0:float ->
                                  beta:float ->
                                  m:float ->
                                  k:float ->
                                  alpha:float ->
                                  lambda:float ->
                                  float ->
                                  float

  val compute_l2_convergence : initial_diff:float ->
                              m:float ->
                              k:float ->
                              alpha:float ->
                              gamma:float ->
                              float ->
                              float
end

(* Interface *)
module McKeanLinearization : sig
  type config = {
    beta: float;
    dimension: int;
    max_iter: int;
    tolerance: float;
    t_final: float;
    dt: float;
  }

  val default_config : config

  val solve : confining_potential:Potential.t ->
              interaction_potential:Potential.t ->
              ?config:config ->
              (Tensor.t -> Tensor.t) ->
              McKeanSDE.t * LinearizedProcess.t * FokkerPlanck.solution
end