open Torch

type t

val create : ?num_factors:int -> ?hidden_size:int -> ?dropout:float -> ?device:Device.t -> unit -> t

val stock_embed : t -> Tensor.t -> Tensor.t -> Tensor.t

val encode : t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t

val decode : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t

val sample_latent : t -> int -> Tensor.t

val loss : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

val train : t -> Optimizer.t -> Tensor.t -> Tensor.t -> Tensor.t -> float

val train_epoch : t -> Optimizer.t -> Data.dataset -> float

val evaluate : t -> Data.dataset -> float

val predict : t -> Tensor.t -> Tensor.t -> Tensor.t

val train_model : t -> Optimizer.t -> int -> Data.dataset -> Data.dataset -> unit

val parameters : t -> Tensor.t list

val named_parameters : t -> (string * Tensor.t) list