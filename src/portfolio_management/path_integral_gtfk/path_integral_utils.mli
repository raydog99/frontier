open Torch

val generate_black_scholes_paths : 
  int -> int -> float -> float -> float -> float -> Tensor.t

val generate_heston_paths : 
  int -> int -> float -> float -> float -> float -> float -> float -> float -> Tensor.t

val generate_sabr_paths : 
  int -> int -> float -> float -> float -> float -> float -> float -> Tensor.t

val calculate_black_scholes_action : Tensor.t -> float -> float -> float -> Tensor.t

val calculate_heston_action : Tensor.t -> float -> float -> float -> float -> float -> float -> Tensor.t

val calculate_sabr_action : Tensor.t -> float -> float -> float -> float -> float -> Tensor.t

val european_call_payoff : float -> Tensor.t -> Tensor.t

val european_put_payoff : float -> Tensor.t -> Tensor.t\

val american_call_payoff : float -> Tensor.t -> Tensor.t

val american_put_payoff : float -> Tensor.t -> Tensor.t

val asian_call_payoff : Tensor.t -> float -> Tensor.t -> Tensor.t

val asian_put_payoff : Tensor.t -> float -> Tensor.t -> Tensor.t

val barrier_call_payoff : Tensor.t -> float -> Tensor.t -> Tensor.t

val barrier_put_payoff : Tensor.t -> float -> Tensor.t -> Tensor.t

val calculate_average_prices : Tensor.t -> int -> Tensor.t

val calculate_barrier_condition : 
  Tensor.t -> 
  [< `UpAndOut | `DownAndOut | `UpAndIn | `DownAndIn ] -> 
  float -> 
  Tensor.t

val discount_factor : 
  [< `BlackScholes of {r: float; sigma: float} 
   | `Heston of {r: float; kappa: float; theta: float; xi: float; rho: float}
   | `SABR of {alpha: float; beta: float; rho: float; nu: float} ] -> 
  float -> 
  float