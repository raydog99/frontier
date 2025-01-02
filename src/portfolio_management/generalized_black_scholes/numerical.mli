open Utils

module FiniteDifference : sig
  type scheme_matrices = {
    a: Tensor.t;
    b: Tensor.t;
    psi: Tensor.t;
  }
  val build_matrices : BlackScholesOperator.params -> Grid.t -> Measure.t -> scheme_matrices
  val solve : GeneralizedBlackScholesSolver.params -> scheme_matrices -> float array -> float -> int -> Tensor.t
end

module Validation : sig
  type error_stats = {
    l2_error: float;
    max_error: float;
    convergence_rate: float;
  }
  val analyze_error : Tensor.t -> float array -> Grid.t -> error_stats
  val check_convergence : error_stats -> float -> bool
end