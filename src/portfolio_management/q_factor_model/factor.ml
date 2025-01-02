open Torch

type t = {
  name: string;
  data: Tensor.t;
}

let create name data =
  { name; data }

let mean factor =
  Tensor.mean factor.data ~dim:[0] ~keepdim:false

let std factor =
  Tensor.std factor.data ~dim:[0] ~keepdim:false ~unbiased:true

let sharpe factor =
  let mean = mean factor in
  let std = std factor in
  Tensor.div mean std

let t_statistic factor =
  let m = mean factor in
  let s = std factor in
  let n = Tensor.size factor.data 0 in
  Util.t_statistic m s n

let p_value factor =
  let t_stat = t_statistic factor in
  let df = Tensor.size factor.data 0 - 1 in
  Util.p_value t_stat (float_of_int df)

let calculate_premium factor =
  let mean_return = mean factor in
  let std_dev = std factor in
  let t_stat = t_statistic factor in
  let p_val = p_value factor in
  (mean_return, std_dev, t_stat, p_val)

let calculate_irr factor =
  Util.calculate_irr factor.data

let calculate_newey_west_t_stat factor lags =
  let mean_return = mean factor in
  let nw_std = Util.newey_west_adjustment factor.data lags in
  let n = Tensor.size factor.data 0 in
  Util.t_statistic mean_return nw_std n

let bootstrap_confidence_interval factor confidence_level n_bootstrap =
  Util.bootstrap_confidence_interval factor.data confidence_level n_bootstrap

let calculate_cumulative_return factor =
  Tensor.cumprod (Tensor.add factor.data (Tensor.float 1.)) ~dim:0

let calculate_drawdown factor =
  let cumulative_return = calculate_cumulative_return factor in
  let running_max = Tensor.cummax cumulative_return ~dim:0 |> fst in
  Tensor.div (Tensor.sub running_max cumulative_return) running_max

let calculate_sortino_ratio factor risk_free_rate =
  let excess_returns = Tensor.sub factor.data risk_free_rate in
  let downside_returns = Tensor.where (Tensor.lt excess_returns (Tensor.float 0.)) excess_returns (Tensor.zeros_like excess_returns) in
  let downside_deviation = Tensor.std downside_returns ~dim:[0] ~unbiased:true in
  Tensor.div (Tensor.mean excess_returns) downside_deviation

let calculate_factor_timing factor market_factor =
  let factor_returns = factor.data in
  let market_returns = market_factor.data in
  let x = Tensor.stack [market_returns; Tensor.mul market_returns factor_returns] ~dim:1 in
  let y = factor_returns in
  let xt = Tensor.transpose x 0 1 in
  let xtx = Tensor.matmul xt x in
  let xty = Tensor.matmul xt y in
  let coeffs = Tensor.solve xtx xty in
  let timing_coeff = Tensor.select coeffs 0 1 in
  let t_stat = Util.t_statistic timing_coeff (Tensor.std timing_coeff ~unbiased:true) (Tensor.size factor_returns 0) in
  (timing_coeff, t_stat)

let calculate_factor_timingseries factor market_factor window_size =
  let factor_returns = factor.data in
  let market_returns = market_factor.data in
  let n = Tensor.size factor_returns 0 in
  let timing_coeffs = Tensor.zeros [n - window_size + 1] in
  let t_stats = Tensor.zeros [n - window_size + 1] in
  for i = 0 to n - window_size do
    let window_factor = Tensor.narrow factor_returns ~dim:0 ~start:i ~length:window_size in
    let window_market = Tensor.narrow market_returns ~dim:0 ~start:i ~length:window_size in
    let (timing_coeff, t_stat) = calculate_factor_timing (create "Window" window_factor) (create "Market" window_market) in
    Tensor.set timing_coeffs [i] timing_coeff;
    Tensor.set t_stats [i] t_stat;
  done;
  (timing_coeffs, t_stats)