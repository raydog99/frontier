open Types

val calculate_returns : portfolio -> Tensor.t
val calculate_expected_return : portfolio -> Tensor.t
val calculate_risk : portfolio -> Tensor.t -> float
val markowitz_optimize : Tensor.t -> Tensor.t -> float -> Tensor.t
val create_portfolio : asset list -> Tensor.t -> portfolio