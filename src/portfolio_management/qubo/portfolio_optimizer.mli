type t
type strategy = SingleStage | TwoStage

val create : int -> float array -> float array array -> t
val optimize : t -> float -> int -> float -> float -> int -> float array
val estimate_penalty_coefficient : t -> float -> int -> float
val two_stage_search : t -> float -> int -> float -> float -> int -> float array
val calculate_portfolio_return : t -> float array -> float
val calculate_portfolio_risk : t -> float array -> float
val calculate_sharpe_ratio : t -> float array -> float -> float
val backtest : t -> strategy -> float -> int -> float -> float -> int -> int -> float array
val compare_strategies : t -> float -> int -> float -> float -> int -> int -> float array * float array
val parameter_sensitivity_analysis : t -> float -> int list -> float list -> float list -> int -> (int * float * float * float) list