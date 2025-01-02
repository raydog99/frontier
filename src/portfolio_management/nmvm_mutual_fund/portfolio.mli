open Torch

type t = Tensor.t
type constraint_type = 
  | NoShortSelling
  | SectorExposure of (int list * float * float) list
  | MaxWeight of float
  | MinWeight of float
  | TurnoverLimit of float

val create : Tensor.t -> t
val expected_wealth : Nmvm.t -> t -> float -> float -> Tensor.t
val variance : Nmvm.t -> t -> Tensor.t
val sharpe_ratio : Nmvm.t -> t -> float -> float
val apply_constraints : t -> constraint_type list -> t
val rebalance : t -> Nmvm.t -> constraint_type list -> t
val turnover : t -> t -> float
val tracking_error : Nmvm.t -> t -> t -> float