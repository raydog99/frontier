open Torch

type t

val create : 
  initial_price:float -> 
  drift:float -> 
  volatility:float -> 
  jump_intensity:float -> 
  jump_size:float -> 
  t

val simulate_price : t -> time_steps:int -> dt:float -> Tensor.t