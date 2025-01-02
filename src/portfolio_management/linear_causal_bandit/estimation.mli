open Torch

module Lasso : sig
  val coordinate_descent : 
    x:Tensor.t -> y:Tensor.t -> lambda:float -> max_iter:int -> Tensor.t
end

module RidgeRegression : sig
  val estimate : x:Tensor.t -> y:Tensor.t -> lambda:float -> Tensor.t
end