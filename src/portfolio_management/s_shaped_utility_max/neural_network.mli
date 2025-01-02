open Torch

module type Network = sig
  val model : nn
end

module type NetworkArchitecture = sig
  val create : int -> int -> int -> (module Network)
end

module FeedForward : NetworkArchitecture
module LSTM : NetworkArchitecture