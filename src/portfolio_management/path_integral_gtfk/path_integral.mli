open Torch

type t
type option_type = Call | Put
type barrier_type = UpAndOut | DownAndOut | UpAndIn | DownAndIn

type model =
  | BlackScholes of { r: float; sigma: float }
  | Heston of { r: float; kappa: float; theta: float; xi: float; rho: float }
  | LocalVol of { r: float; sigma: Tensor.t -> float -> float }

val create :
  model:model ->
  num_time_steps:int ->
  num_paths:int ->
  maturity:float ->
  initial_state:Tensor.t ->
  t

val price_european_option : t -> strike:float -> option_type:option_type -> Tensor.t
val price_american_option : t -> strike:float -> option_type:option_type -> Tensor.t
val price_barrier_option : t -> strike:float -> option_type:option_type -> barrier_type:barrier_type -> barrier_level:float -> Tensor.t
val price_asian_option : t -> strike:float -> option_type:option_type -> averaging_points:int -> Tensor.t

val calculate_greeks : t -> strike:float -> option_type:option_type -> float * float * float * float

val parallel_price_european_option : t -> strike:float -> option_type:option_type -> Tensor.t
val parallel_price_american_option : t -> strike:float -> option_type:option_type -> Tensor.t