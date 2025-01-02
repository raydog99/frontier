open Torch

val calculate_r_squared : Tensor.t -> Tensor.t -> Tensor.t
val calculate_adjusted_r_squared : float -> int -> int -> float
val calculate_sharpe_ratio : Tensor.t -> Tensor.t -> Tensor.t
val calculate_morningstar_risk_adjusted_return : Tensor.t -> Tensor.t -> float -> int -> Tensor.t
val calculate_stress_var : Tensor.t -> Tensor.t -> float -> Tensor.t
val calculate_long_term_alpha : Tensor.t -> Tensor.t -> Tensor.t
val calculate_long_term_ratio : Tensor.t -> Tensor.t -> Tensor.t
val calculate_long_term_stability : Tensor.t -> Tensor.t -> float -> Tensor.t
val polynomial_regression : Tensor.t -> Tensor.t -> int -> Tensor.t
val target_shuffling : Tensor.t -> Tensor.t -> int -> float
val extract_features : Tensor.t -> Tensor.t -> Tensor.t
val extract_features_sequence : Tensor.t -> Tensor.t -> int -> Tensor.t