open Torch

type t = {
  self_attention: Nn.t;
  feedforward: Nn.t;
  layer_norm1: Nn.t;
  layer_norm2: Nn.t;
}

let create dim num_heads =
  let self_attention = Nn.multi_head_attention dim num_heads in
  let feedforward = Nn.sequential [
    Nn.linear dim (4 * dim);
    Nn.relu ();
    Nn.linear (4 * dim) dim;
  ] in
  let layer_norm1 = Nn.layer_norm [dim] in
  let layer_norm2 = Nn.layer_norm [dim] in
  { self_attention; feedforward; layer_norm1; layer_norm2 }

let forward t x =
  let attn_output = Nn.apply t.self_attention x x x in
  let x = Nn.apply t.layer_norm1 (Tensor.(x + attn_output)) in
  let ff_output = Nn.apply t.feedforward x in
  Nn.apply t.layer_norm2 (Tensor.(x + ff_output))

let state_dict t =
  [("self_attention", Nn.state_dict t.self_attention);
   ("feedforward", Nn.state_dict t.feedforward);
   ("layer_norm1", Nn.state_dict t.layer_norm1);
   ("layer_norm2", Nn.state_dict t.layer_norm2)]

let load_state_dict t state_dict =
  Nn.load_state_dict t.self_attention (List.assoc "self_attention" state_dict);
  Nn.load_state_dict t.feedforward (List.assoc "feedforward" state_dict);
  Nn.load_state_dict t.layer_norm1 (List.assoc "layer_norm1" state_dict);
  Nn.load_state_dict t.layer_norm2 (List.assoc "layer_norm2" state_dict)