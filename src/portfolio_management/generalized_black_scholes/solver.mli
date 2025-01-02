open Utils
open Pricing

module BlackScholesOperator : sig
  type params = {
    r: float;
    sigma: float;
  }
  
  val make : float -> float -> params
  val apply : params -> Grid.t -> float array -> float array
  val check_coercivity : params -> bool
end

module GeneralizedBlackScholesSolver : sig
  type params = {
    measure: Measure.t;
    bs_params: BlackScholesOperator.params;
    grid: Grid.t;
    dt: float;
    n_steps: int;
    strike: float;
    option_type: OptionType.t;
    scheme: NumericalScheme.scheme;
  }

  val make_payoff : params -> Grid.t -> float array
  val solve : params -> (Tensor.t * Greeks.t)
  val validate : params -> Tensor.t -> bool
end