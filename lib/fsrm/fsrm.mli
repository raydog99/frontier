open Torch

type t

val create : hurst:float -> eta:float -> kappa:float -> scale:float -> t
val sample : t -> int -> Tensor.t
val regularity : t -> Tensor.t -> Tensor.t
val binarize : Tensor.t -> float -> Tensor.t
val serial_information : Tensor.t -> int -> float
val theoretical_serial_information : t -> float -> float
val plot_sample : t -> int -> string -> unit
val plot_regularity : t -> Tensor.t -> string -> unit
val local_regularity : t -> Tensor.t -> float -> Tensor.t
val multifractal_spectrum : t -> Tensor.t -> Tensor.t * Tensor.t
val residuals : t -> Tensor.t -> Tensor.t
val standardized_residuals : t -> Tensor.t -> Tensor.t