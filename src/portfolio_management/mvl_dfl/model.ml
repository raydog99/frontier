open Torch

type t = {
    layers : Nn.t;
  }

let create input_size hidden_size output_size =
    let layers = Nn.sequential [
      Nn.linear input_size hidden_size ~bias:true;
      Nn.relu ();
      Nn.linear hidden_size output_size ~bias:true;
    ] in
    { layers }

let forward model x =
    Nn.forward model.layers x

let parameters model =
    Nn.parameters model.layers