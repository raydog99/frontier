open Utils

module OptionType : sig
  type t = Call | Put
  val payoff : t -> float -> float -> float
  val boundary_condition : t -> float -> float -> float -> float -> float
end

module Greeks : sig
  type t = {
    delta: Tensor.t;
    gamma: Tensor.t;
    theta: Tensor.t;
    vega: Tensor.t;
    rho: Tensor.t;
  }
  val calculate : BlackScholesOperator.params -> Grid.t -> float -> Tensor.t -> t
end