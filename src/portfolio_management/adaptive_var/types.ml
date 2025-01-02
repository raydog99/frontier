open Torch

type loss_function = Tensor.t -> Tensor.t -> Tensor.t

type framework = 
  | Lipschitz_gradient
  | Holder_density
  | Subexponential_deviation

type error =
  | InvalidParameter of string
  | ComputationError of string
  | DistributedComputingError of string
  | IOError of string
  | UnsupportedFeature of string

exception Error of error