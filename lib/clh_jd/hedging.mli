open Torch

type hedging_strategy

type hedging_method = CLH | CMH

val create : 
  asset:Asset.t -> 
  option:Option.t -> 
  transaction_cost:float -> 
  time_steps:int -> 
  dt:float -> 
  hedging_strategy

val conditional_expectation_f : 
  hedging_strategy -> Tensor.t -> Tensor.t -> Tensor.t

val conditional_expectation_s_f : 
  hedging_strategy -> Tensor.t -> Tensor.t -> Tensor.t

val calculate_u : 
  hedging_strategy -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

val hedge :
  hedging_strategy -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> hedging_method -> Tensor.t