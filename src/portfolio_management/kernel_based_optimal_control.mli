open Torch

(** Core types *)
type state = Tensor.t
type control = Tensor.t
type time = float

module Dataset : sig
  type t = {
    states: Tensor.t;           (** N x nx matrix of state samples *)
    derivatives: Tensor.t;      (** N x nx matrix of state derivatives *)
    controls: Tensor.t option;  (** N x nu matrix of control inputs if available *)
  }

  val create : states:Tensor.t -> derivatives:Tensor.t -> ?controls:Tensor.t -> unit -> t
end

module Measure : sig
  type t

  val create : density:(Tensor.t -> float Tensor.t) -> 
              support:(float * float) array -> t
  
  val in_support : t -> Tensor.t -> bool
  val evaluate : t -> Tensor.t -> float Tensor.t
end

(** Kernel functions *)
val gaussian_k : sigma:float -> Tensor.t -> Tensor.t -> float Tensor.t
val gaussian_grad_k : sigma:float -> Tensor.t -> Tensor.t -> Tensor.t
val gaussian_hessian_k : sigma:float -> Tensor.t -> Tensor.t -> Tensor.t
val third_derivative_k : sigma:float -> Tensor.t -> Tensor.t -> Tensor.t
val mixed_derivatives_k : sigma:float -> Tensor.t -> Tensor.t -> Tensor.t array

(** System dynamics *)
module Dynamics : sig
  type t = {
    nx: int;   (** State dimension *)
    nu: int;   (** Control dimension *)
    drift: Tensor.t -> Tensor.t;  (** Drift function f(x) *)
    diffusion: float;  (** Diffusion coefficient epsilon *)
  }

  val create : nx:int -> nu:int -> drift:(Tensor.t -> Tensor.t) -> diffusion:float -> t
end

(** Control functions *)
val control_matrix : Tensor.t -> Tensor.t  (** Control matrix G(x) *)
val controlled_drift : Tensor.t -> control -> Tensor.t  (** Full dynamics f(x) + G(x)u *)

(** Cost functions *)
module Cost : sig
  type t = {
    stage_cost: Tensor.t -> float Tensor.t;  (** Running cost q(x) *)
    control_penalty: Tensor.t -> float Tensor.t;  (** Control cost r(u) *)
  }

  val create : stage_cost:(Tensor.t -> float Tensor.t) -> 
               control_penalty:(Tensor.t -> float Tensor.t) -> t
end

(** Operator theory *)
module HilbertOperator : sig
  type t

  val create : forward:(Tensor.t -> Tensor.t) -> 
               adjoint:(Tensor.t -> Tensor.t) -> 
               domain_dim:int -> range_dim:int -> t

  val compose : t -> t -> t  (** Operator composition *)
end

module GeneratorOperator : sig
  type t

  val create : epsilon:float -> t
  val forward : t -> Tensor.t -> Tensor.t  (** Forward generator action *)
  val adjoint : t -> Tensor.t -> Tensor.t  (** Adjoint generator action *)
end

(** Fokker-Planck-Kolmogorov equation *)
module FPK : sig
  type t

  val create : dynamics:Dynamics.t -> epsilon:float -> t
  val forward_evolution : t -> Tensor.t -> float -> Tensor.t  (** Forward FPK flow *)
  val backward_evolution : t -> Tensor.t -> float -> Tensor.t  (** Backward FPK flow *)
end

(** Hamilton-Jacobi-Bellman equation *)
module HJB : sig
  type t

  val create : fpk:FPK.t -> cost:Cost.t -> final_time:float -> dt:float -> t
  
  (** Solve HJB equation returning (value_function, density) *)
  val solve : t -> Tensor.t -> Tensor.t * Tensor.t
end

(** Generator regression *)
module GeneratorRegression : sig
  type t

  val create : epsilon:float -> reg_param:float -> t
  
  (** Learn autonomous generator from data *)
  val learn_generator : t -> Dataset.t -> Tensor.t -> Tensor.t

  (** Learn controlled generator system 
      Returns (autonomous_generator, control_generators) *)
  val learn_controlled_generator : t -> Dataset.t -> 
    (Tensor.t -> Tensor.t) * (Tensor.t -> Tensor.t array)
end

(** Duality and measure *)
module DualityMeasure : sig
  type dual_pair

  val create_dual_pair : primal:Measure.t -> 
                        dual:Measure.t -> 
                        pairing:(Tensor.t -> Tensor.t -> float Tensor.t) -> 
                        dual_pair

  (** Verify strong duality conditions *)
  val verify_strong_duality : dual_pair -> (unit, string) result

  (** Compute primal value, dual value and duality gap *)
  val optimal_values : dual_pair -> Tensor.t -> Tensor.t -> 
                      float Tensor.t * float Tensor.t * float Tensor.t
end