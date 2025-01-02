open Torch

(* Numerical stability utilities *)
val stable_log : Tensor.t -> Tensor.t
val logsumexp : Tensor.t -> Tensor.t
val log_normalize : Tensor.t -> Tensor.t
val normalize : Tensor.t -> Tensor.t

(* Matrix utilities *)
val is_pos_def : Tensor.t -> bool
val ensure_pos_def : Tensor.t -> Tensor.t

(* Distribution utilities *)
val kl_divergence : Tensor.t -> Tensor.t -> Tensor.t
val mvnormal : Tensor.t -> Tensor.t -> Tensor.t

(* Resampling utilities *)
val systematic_resample : Tensor.t -> int -> Tensor.t