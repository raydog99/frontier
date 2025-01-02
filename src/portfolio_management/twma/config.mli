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

val create : ?initial_cash:float -> ?window_size:int -> ?alpha:float ->
             ?image_size:int * int -> ?transaction_cost:float ->
             ?learning_rate:float -> ?num_folds:int -> ?epochs:int ->
             ?num_threads:int -> ?portfolio_method:Portfolio_constructor.method_t ->
             ?assets:string array -> ?in_sample_periods:int ->
             ?out_sample_periods:int -> ?max_drawdown:float ->
             ?var_threshold:float -> ?volatility_target:float ->
             ?max_leverage:float -> ?fundamental_data_file:string ->
             ?use_ml:bool -> ?slippage:float -> unit -> t

val load_from_file : string -> t
val save_to_file : t -> string -> unit