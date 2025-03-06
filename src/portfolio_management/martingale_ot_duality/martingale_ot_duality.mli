open Torch

(* Skorokhod space representation: Rd-valued cadlag functions *)
module Skorokhod : sig
  (* Cadlag function type representing elements of the Skorokhod space *)
  type t
  
  (* Create a function from time points and values *)
  val of_points : float array -> Tensor.t array -> t
  
  (* Evaluate at time t *)
  val eval : t -> float -> Tensor.t
  
  (* Get supremum norm *)
  val sup_norm : t -> float
  
  (* Get path supremum *)
  val path_supremum : t -> Tensor.t
  
  (* Count up-crossings of a function from level a to level b *)
  val up_crossings : t -> float -> float -> int
  
  (* Compute the integral of a function over an interval *)
  val integrate : t -> float -> float -> Tensor.t
  
  (* Find index of the largest time less than or equal to t *)
  val find_index : t -> float -> int
end

(* Measure space and probability measures *)
module Measure : sig
  (* Type representing a Radon probability measure *)
  type t
  
  (* Create a probability measure from a density function *)
  val create : (Skorokhod.t -> float) -> t
  
  (* Create a measure from sample paths with weights *)
  val create_from_samples : Skorokhod.t array -> float array -> t
  
  (* Generate sample paths for Monte Carlo estimation *)
  val generate_samples : int -> Skorokhod.t array
  
  (* Compute expectation of a function under the measure *)
  val expectation : t -> (Skorokhod.t -> float) -> float
  
  (* Check if measure satisfies EQ[gamma] <= 0 for all gamma in a set *)
  val satisfies_constraints : t -> (Skorokhod.t -> float) list -> bool
  
  (* Create a single martingale measure from empirical data *)
  val create_empirical_martingale : Skorokhod.t array -> t
  
  (* Create a collection of martingale measures satisfying constraints *)
  val create_q_g : (Skorokhod.t -> float) list -> t list
end

(* Riesz spaces *)
module Riesz : sig
  (* Type representing functions in Riesz spaces *)
  type t
  
  (* Create from a function *)
  val of_function : (Skorokhod.t -> float) -> t
  
  (* Create from a function with specific space type *)
  val of_function_in_space : (Skorokhod.t -> float) -> 
                            [ `Continuous | `BoundedBorel | `UpperSemiContinuous | `BorelP ] -> t
  
  (* Create from a function in B_p(Ω) space *)
  val of_function_in_bp : (Skorokhod.t -> float) -> float -> t
  
  (* Apply the function *)
  val apply : t -> Skorokhod.t -> float
  
  (* Check if a function is bounded over a set of sample paths *)
  val is_bounded_over_paths : t -> Skorokhod.t array -> bool
  
  (* Check boundedness *)
  val is_bounded : t -> bool
  
  (* Basic operations *)
  val add : t -> t -> t
  val mul : float -> t -> t
  val max : t -> t -> t
  val min : t -> t -> t
  
  (* Create indicator function of a set *)
  val indicator_function : (Skorokhod.t -> bool) -> t
  
  (* Create a truncated version of a function *)
  val truncate : t -> float -> t
  
  (* Create the positive part of a function: ξ⁺ = max(ξ, 0) *)
  val positive_part : t -> t
  
  (* Create the negative part of a function: ξ⁻ = max(-ξ, 0) *)
  val negative_part : t -> t
end

(* Integrals and quotient sets *)
module Integrand : sig
  (* Simple integrand type *)
  type simple_t = {
    stopping_times: (Skorokhod.t -> float) array;  (* tau_n stopping times *)
    values: (Skorokhod.t -> Tensor.t) array;       (* h_n predictable processes *)
  }
  
  (* General integrand type: sequence of simple integrands *)
  type t = simple_t array
  
  (* Find the lambda bound for admissibility *)
  val find_lambda_bound : simple_t -> Skorokhod.t array -> float -> float
  
  (* Compute the stochastic integral (H·X)_t for a simple integrand *)
  val integrate_simple : simple_t -> Skorokhod.t -> float -> float
  
  (* Compute the stochastic integral (H·X)_t for a general integrand *)
  val integrate : t -> Skorokhod.t -> float -> float
  
  (* Create the quotient set I_s(G) *)
  val create_quotient_is : simple_t list -> (Skorokhod.t -> float) list -> (Skorokhod.t -> float) list
  
  (* Create the quotient set I(0) *)
  val create_quotient_i0 : t list -> (Skorokhod.t -> float) list
  
  (* Create the quotient set I(G) *)
  val create_quotient_ig : t list -> (Skorokhod.t -> float) list -> (Skorokhod.t -> float) list
  
  (* Create the Fatou-closure of I(G) *)
  val create_fatou_closure : (Skorokhod.t -> float) list -> float -> (Skorokhod.t -> float) list
end