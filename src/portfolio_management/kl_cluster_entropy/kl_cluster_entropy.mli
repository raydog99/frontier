open Torch

type time_series = Tensor.t

type cluster = {
  start_index: int;
  end_index: int;
  duration: int;
}

type probability_distribution = (int * float) list

type market_data = {
  asset_name: string;
  prices: time_series;
  returns: time_series;
  volatility: time_series;
}

val moving_average : time_series -> int -> time_series

val generate_clusters : time_series -> int -> cluster list

val calculate_cluster_probabilities : cluster list -> probability_distribution

val kullback_leibler_divergence : probability_distribution -> probability_distribution -> float

val kl_cluster_entropy : time_series -> time_series -> int -> int -> float

val shannon_cluster_entropy : time_series -> int -> int -> float

val realized_volatility : time_series -> int -> time_series

val process_market_data : string -> int -> market_data

val kl_cluster_weights : market_data list -> int -> int -> float list

val optimal_portfolio : market_data list -> int -> int -> int -> float array

val load_multiple_assets : string list -> int -> market_data list

val portfolio_statistics : float array -> float * float * float