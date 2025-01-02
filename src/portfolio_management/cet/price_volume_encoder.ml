open Torch

type t = {
  encoder: Nn.t;
}

let create input_dim hidden_dim =
  let encoder = Nn.sequential [
    Nn.linear input_dim hidden_dim;
    Nn.relu ();
  ] in
  { encoder }

let forward t x =
  Nn.apply t.encoder x

let state_dict t = Nn.state_dict t.encoder
let load_state_dict t state_dict = Nn.load_state_dict t.encoder state_dict