open Torch

(* Node representation *)
module Node : sig
  type t
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val hash : t -> int
  val to_string : t -> string
end

module NodeSet : Set.S with type elt = Node.t
module NodeMap : Map.S with type key = Node.t

(* Edge representation *)
module Edge : sig
  type t
  val create : Node.t -> Node.t -> t
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val hash : t -> int
end

module EdgeSet : Set.S with type elt = Edge.t
module EdgeMap : Map.S with type key = Edge.t

(* Stratum representation *)
module Stratum : sig
  type condition = {
    node : Node.t;
    lower_bound : float;
    upper_bound : float;
  }

  type t = condition list

  val create : Node.t list -> (float * float) list -> t
  val is_satisfied : t -> Tensor.t -> bool
end

(* Graph representation *)
module Graph : sig
  type t = {
    nodes : NodeSet.t;
    edges : EdgeSet.t;
    strata : (Edge.t * Stratum.t list) list;
  }

  val create : Node.t list -> Edge.t list -> (Edge.t * Stratum.t list) list -> t
  val adjacent : t -> Node.t -> NodeSet.t
  val find_path : t -> Node.t -> Node.t -> Node.t list option
  val is_decomposable : t -> bool
end

(* Numerical optimization *)
module NumericalOpt : sig
  module Stabilization : sig
    val stable_inverse : Tensor.t -> Tensor.t
    val stable_cholesky : Tensor.t -> Tensor.t
    val stable_logdet : Tensor.t -> float
  end
end

(* Multivariate Gaussian distribution *)
module MultivariateGaussian : sig
  type t = {
    mean : Tensor.t;
    covariance : Tensor.t;
    precision : Tensor.t option;
    cholesky : Tensor.t option;
    dim : int;
  }

  val create : Tensor.t -> Tensor.t -> t
  val log_pdf : t -> Tensor.t -> float
  val sample : t -> int -> Tensor.t
end

(* Piecewise distribution *)
module PiecewiseDistribution : sig
  type piece = {
    region : Stratum.t;
    distribution : MultivariateGaussian.t;
  }

  type t = {
    pieces : piece list;
    dim : int;
  }

  val create : piece list -> int -> t
  val log_pdf : t -> Tensor.t -> float
  val sample : t -> int -> Tensor.t
end

(* Core SGGM functionality *)
module SGGM : sig
  type t = {
    graph : Graph.t;
    dim : int;
    sigma : Tensor.t;
  }

  val create : Graph.t -> int -> t
  val mle : t -> Tensor.t -> t
  val log_likelihood : t -> Tensor.t -> float
  val score : t -> Tensor.t -> float
  val prior : t -> float
end

(* Clique finding algorithms *)
module CompleteClique : sig
  val find_maximal_cliques : Graph.t -> NodeSet.t list
  val find_maximum_clique : Graph.t -> Node.t list
end

(* Separator algorithms *)
module CompleteSeparator : sig
  val find_minimal_separators : Graph.t -> NodeSet.t list
  val find_minimum_weight_separator : Graph.t -> float array -> (NodeSet.t * float) option
end

(* Model selection *)
module ModelSelection : sig
  type mcmc_state = {
    model : SGGM.t;
    score : float;
  }

  type proposal_type =
    | AddEdge
    | RemoveEdge
    | AddStratum
    | RemoveStratum
    | ModifyStratum
    | SwapEdges
    | SplitStratum
    | MergeStrata
    | FlipEdge
    | ModifyBoundary

  val generate_stratum : Node.t list -> Stratum.t
  val propose_next : mcmc_state -> SGGM.t
  val count_parameters : SGGM.t -> int
  val search : SGGM.t -> Tensor.t -> int -> SGGM.t
end

(* Context-specific independence verification *)
module CSIndependence : sig
  type independence_check = {
    vars : Node.t list;
    context : Stratum.t;
    is_independent : bool;
  }

  val verify_conditional_independence : SGGM.t -> Tensor.t -> Node.t list -> Stratum.t -> bool
  val verify_all_independencies : SGGM.t -> Tensor.t -> independence_check list
end

(* Curved exponential family verification *)
module CurvedFamily : sig
  type curved_family_check = {
    smooth_density : bool;
    continuous_statistics : bool;
    parameter_constraints : bool;
    exponential_family : bool;
  }

  val verify_smooth_density : SGGM.t -> Tensor.t -> bool
  val verify_parameter_constraints : SGGM.t -> bool
  val verify : SGGM.t -> Tensor.t -> curved_family_check
end

(* Normalizing constant computation *)
module NormalizingConstant : sig
  type integration_method =
    | MonteCarlo of int
    | ImportanceSampling of int
    | LaplaceMean

  val compute : integration_method -> SGGM.t -> float
end