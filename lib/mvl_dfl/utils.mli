open Torch

val device : Device.t
val to_device : Tensor.t -> Tensor.t
val read_csv : string -> (string list list, string) result
val to_tensor : string list list -> (Tensor.t, string) result
val split_data : 'a list -> float -> ('a list * 'a list, string) result
val evaluate_sharpe_ratio : Tensor.t -> Tensor.t -> Tensor.t -> (Tensor.t, string) result
val calculate_covariance : Tensor.t -> (Tensor.t, string) result