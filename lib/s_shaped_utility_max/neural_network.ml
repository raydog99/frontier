open Torch

module type Network = sig
  val model : nn
end

module type NetworkArchitecture = sig
  val create : int -> int -> int -> (module Network)
end

module FeedForward = struct
  let create input_dim hidden_dim output_dim =
    let module N = struct
      let model = 
        Sequential.([
          linear ~in_features:input_dim ~out_features:hidden_dim ();
          relu ();
          linear ~in_features:hidden_dim ~out_features:hidden_dim ();
          relu ();
          linear ~in_features:hidden_dim ~out_features:output_dim ();
        ])
    end in
    (module N : Network)
end

module LSTM = struct
  let create input_dim hidden_dim output_dim =
    let module N = struct
      let model =
        let lstm = LSTM.create ~input_dim ~hidden_size:hidden_dim () in
        let linear = Linear.create ~in_features:hidden_dim ~out_features:output_dim () in
        fun x ->
          let _, (h, _) = LSTM.seq lstm x in
          Linear.apply linear (Tensor.select h (-1) (-1))
    end in
    (module N : Network)
end