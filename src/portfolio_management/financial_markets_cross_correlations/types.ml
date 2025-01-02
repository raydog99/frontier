type market_index = {
  name: string;
  prices: float array;
}

type dcca_result = {
  coefficient: float;
  distance: float;
}

type timescale = Short | Long

type dccc_result = {
  short_scale: int;
  long_scale: int;
  dccc: float;
  timestamp: float;
}

type analysis_result = {
  dccc_results: dccc_result array;
  short_msts: float array array array;
  long_msts: float array array array;
  timestamps: float array;
}

type analysis_config = {
  window_size: int;
  step: int;
  short_timescale: int;
  long_timescale: int;
}

type regime = Bull | Bear | Neutral

type regime_detection_result = {
  timestamp: float;
  regime: regime;
}

type market_efficiency = {
  name: string;
  hurst_exponent: float;
}

type volatility_cluster = {
  start_timestamp: float;
  end_timestamp: float;
  intensity: float;
}

type analysis_summary = {
  dccc_results: dccc_result array;
  regime_results: regime_detection_result array;
  influential_markets: (string * float) list;
  market_efficiencies: market_efficiency array;
  volatility_clusters: volatility_cluster array;
}