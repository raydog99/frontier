open Torch

type t = {
  encoder: Nn.t;
  decoder: Nn.t;
}

let create input_dim hidden_dim =
  let encoder = Nn.sequential [
    Nn.linear input_dim hidden_dim;
    Nn.relu ();
  ] in
  let decoder = Nn.sequential [
    Nn.linear hidden_dim input_dim;
    Nn.relu ();
  ] in
  { encoder; decoder }

let forward t x =
  let encoded = Nn.apply t.encoder x in
  let decoded = Nn.apply t.decoder encoded in
  (encoded, decoded)

let state_dict t = 
  [("encoder", Nn.state_dict t.encoder);
   ("decoder", Nn.state_dict t.decoder)]

let load_state_dict t state_dict =
  Nn.load_state_dict t.encoder (List.assoc "encoder" state_dict);
  Nn.load_state_dict t.decoder (List.assoc "decoder" state_dict)