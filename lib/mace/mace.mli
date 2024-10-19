open Torch

module Make : functor (P : Mace_params.MACE_PARAMS) -> sig
  type portfolio
  type model

  val create_portfolio : Tensor.t -> Tensor.t -> portfolio
  val create_model : (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> model
  val train_mace : portfolio -> Tensor.t -> model
  val predict : model -> Tensor.t -> Tensor.t
  val mean_variance_optimization : Tensor.t -> float -> Tensor.t
  val compare_mace_vs_single_stock : Tensor.t -> Tensor.t -> float * float
  val hyperparameter_tuning : Tensor.t -> Tensor.t -> float * float * int * float
  val nonlinear_mean_reversion_machine : portfolio -> Tensor.t -> Tensor.t
  val daily_trading_strategy : model -> portfolio -> Tensor.t -> float -> Tensor.t * Tensor.t
  val loose_bag_mace : portfolio -> Tensor.t -> model
  val factor_based_analysis : Tensor.t -> Tensor.t -> float * float array * float
  val calculate_sharpe_ratio : Tensor.t -> float -> float
  val calculate_max_drawdown : Tensor.t -> float
  val calculate_omega_ratio : Tensor.t -> float -> float
  val run_mace_strategy : portfolio -> Tensor.t -> Tensor.t * Tensor.t * float * float * (float * float) * (float * float array * float)
end