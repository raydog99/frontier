module type MACE_PARAMS = sig
  val learning_rate : float
  val max_iterations : int
  val ridge_lambda : float
  val num_trees : int
  val min_samples_leaf : int
  val mtry : float
  val early_stopping_rounds : int
  val validation_split : float
  val marx_lags : int
  val portfolio_size : int
  val rebalance_frequency : int
  val transaction_cost : float
  val risk_aversion : float
  val loose_bag_r_squared_target : float
  val num_factors : int
end