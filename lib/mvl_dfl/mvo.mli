open Torch

type optimization_method = 
	| QuadraticProgramming
	| GradientDescent
	| SLSQP

val optimize : ?method_:optimization_method -> Tensor.t -> Tensor.t -> float -> Tensor.t
val sharpe_ratio : Tensor.t -> Tensor.t -> Tensor.t