open Torch

module ModelParams : sig
  type t = private {
    d: int;              (** Dimension *)
    l_bar: float;        (** Lower bound on second moment *)
    l: float;            (** Upper bound on features *)
    r: float;            (** Radius of weight vectors *)
    eps: float;          (** Error parameter *)
    eta: float;          (** Learning rate *)
  }

  val create : d:int -> l_bar:float -> l:float -> r:float -> 
               eps:float -> eta:float -> t

  val scale_invariant : t -> t
end

module type LinkFunction = sig
  type t
  
  val create : ?alpha:float -> ?beta:float -> unit -> t

  val apply : t -> Tensor.t -> Tensor.t

  val inverse : t -> Tensor.t -> Tensor.t

  val derivative : t -> Tensor.t -> Tensor.t

  val is_lipschitz : t -> float -> bool

  val is_anti_lipschitz : t -> bool

  val integrate : t -> Tensor.t -> Tensor.t -> Tensor.t
end

module LogisticLink : LinkFunction
module ReLULink : LinkFunction

module DivergenceMeasures : sig
  type divergence = private {
    compute: Tensor.t -> Tensor.t -> Tensor.t;
    gradient: Tensor.t -> Tensor.t -> Tensor.t;
    is_proper: bool;
  }

  val matching_divergence : LinkFunction.t -> divergence

  val proper_divergence : LinkFunction.t -> divergence

  val omnigap_divergence : pred_link:LinkFunction.t -> target_link:LinkFunction.t -> divergence
end

module IsotonicRegression : sig
  val solve : Tensor.t -> Tensor.t

  val fit_link : LinkFunction.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
end

module SIM (P : sig val params : ModelParams.t end) : sig
  type t

  val create : unit -> t

  val predict : t -> Tensor.t -> Tensor.t

  val divergence : t -> Tensor.t -> Tensor.t -> Tensor.t

  val update : ?use_momentum:bool -> t -> Tensor.t -> t
end

module Isotron (P : sig val params : ModelParams.t end) : sig
  type t

  val create : unit -> t

  val step : t -> Tensor.t -> Tensor.t -> t

  val train : ?max_iter:int -> t -> Tensor.t -> Tensor.t -> t

  val predict : t -> Tensor.t -> Tensor.t
end

module MultiIndexModel (P : sig val params : ModelParams.t end) : sig
  type t
  type predictor = private {
    weights: Tensor.t;
    link: LogisticLink.t;
    iteration: int;
  }

  val create : ?max_models:int -> unit -> t

  val add_predictor : t -> Tensor.t -> t

  val predict : t -> Tensor.t -> LinkFunction.t -> Tensor.t
end

module Omnitron (P : sig val params : ModelParams.t end) : sig
  type t
  type stats = private {
    empirical_error: float;
    population_error: float option;
    sample_complexity: int;
    iteration_complexity: int;
  }

  val create : ?max_models:int -> unit -> t

  val step : t -> Tensor.t -> Tensor.t -> t

  val train : ?max_iter:int -> t -> Tensor.t -> Tensor.t -> t

  val predict : t -> Tensor.t -> LinkFunction.t -> Tensor.t

  val omnigap : t -> Tensor.t -> Tensor.t -> LinkFunction.t -> Tensor.t -> Tensor.t
end