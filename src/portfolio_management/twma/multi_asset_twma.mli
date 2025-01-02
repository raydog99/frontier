open Fundamental_data_provider

type t

val create : Config.t -> Asset.t array -> Fundamental_data_provider.t -> t
val update : t -> float array -> unit
val get_portfolio_value : t -> float array -> float
val trade : t -> float array -> float
val train_ml_model : t -> unit