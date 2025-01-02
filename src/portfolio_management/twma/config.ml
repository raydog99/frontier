open Portfolio_constructor

type t = {
  initial_cash: float;
  window_size: int;
  alpha: float;
  image_size: int * int;
  transaction_cost: float;
  learning_rate: float;
  num_folds: int;
  epochs: int;
  num_threads: int;
  portfolio_method: Portfolio_constructor.method_t;
  assets: string array;
  in_sample_periods: int;
  out_sample_periods: int;
  max_drawdown: float;
  var_threshold: float;
  volatility_target: float;
  max_leverage: float;
  fundamental_data_file: string;
  use_ml: bool;
  slippage: float;
}

let create ?(initial_cash=100000.) ?(window_size=50) ?(alpha=0.1) ?(image_size=(50, 50))
           ?(transaction_cost=0.001) ?(learning_rate=0.001) ?(num_folds=5)
           ?(epochs=100) ?(num_threads=4) ?(portfolio_method=Portfolio_constructor.EqualWeight)
           ?(assets=[|"AAPL"; "GOOGL"; "MSFT"; "AMZN"|]) ?(in_sample_periods=252)
           ?(out_sample_periods=63) ?(max_drawdown=0.2) ?(var_threshold=0.05)
           ?(volatility_target=0.15) ?(max_leverage=2.0)
           ?(fundamental_data_file="fundamental_data.csv")
           ?(use_ml=false) ?(slippage=0.0005) () =
  {
    initial_cash; window_size; alpha; image_size; transaction_cost; learning_rate;
    num_folds; epochs; num_threads; portfolio_method; assets; in_sample_periods;
    out_sample_periods; max_drawdown; var_threshold; volatility_target; max_leverage;
    fundamental_data_file; use_ml; slippage;
  }

let load_from_file filename =
  let ic = open_in filename in
  let config = Marshal.from_channel ic in
  close_in ic;
  config

let save_to_file config filename =
  let oc = open_out filename in
  Marshal.to_channel oc config [];
  close_out oc