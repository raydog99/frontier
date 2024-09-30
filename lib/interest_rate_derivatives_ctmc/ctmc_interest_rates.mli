open Torch
open Ctmc

val price_caplet : Ctmc.t -> float -> float -> float -> float -> int -> Tensor.t
val price_floorlet : Ctmc.t -> float -> float -> float -> float -> int -> Tensor.t
val compute_yield_curve_all_states : Ctmc.t -> float -> float -> float -> Tensor.t * Tensor.t
val limiting_yield : Ctmc.t -> Tensor.t
val price_cap : Ctmc.t -> float -> float -> float -> float -> int -> int -> Tensor.t
val price_floor : Ctmc.t -> float -> float -> float -> float -> int -> int -> Tensor.t