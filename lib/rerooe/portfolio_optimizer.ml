type optimization_method =
  | MeanVariance
  | MinimumVariance
  | MaximumSharpeRatio
  | RiskParity

let optimize portfolio method_ =
  let returns = Portfolio.get_returns portfolio in
  let weights = Portfolio.get_weights portfolio in
  match method_ with
  | MeanVariance ->
      let cov_matrix = Statistics.covariance_matrix returns in
      let expected_returns = Statistics.mean returns in
      Optimization.mean_variance cov_matrix expected_returns
  | MinimumVariance ->
      let cov_matrix = Statistics.covariance_matrix returns in
      Optimization.minimum_variance cov_matrix
  | MaximumSharpeRatio ->
      let cov_matrix = Statistics.covariance_matrix returns in
      let expected_returns = Statistics.mean returns in
      let risk_free_rate = 0.02 in
      Optimization.maximum_sharpe_ratio cov_matrix expected_returns risk_free_rate
  | RiskParity ->
      let cov_matrix = Statistics.covariance_matrix returns in
      Optimization.risk_parity cov_matrix