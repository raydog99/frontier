open Torch

type t = {
  lbl: Nn.t;
}

let create hidden_dim =
  let lbl = Nn.linear hidden_dim hidden_dim in
  { lbl }

let forward t c_t z_pos z_neg =
  let f_k_pos = Tensor.(mm (Nn.apply t.lbl c_t) (transpose z_pos ~dim0:0 ~dim1:1)) in
  let f_k_neg = Tensor.(mm (Nn.apply t.lbl c_t) (transpose z_neg ~dim0:0 ~dim1:1)) in
  (f_k_pos, f_k_neg)

let infonce_loss f_k_pos f_k_neg temperature =
  let numerator = Tensor.exp (Tensor.div_scalar f_k_pos temperature) in
  let denominator = Tensor.(numerator + sum (exp (div_scalar f_k_neg temperature)) ~dim:[1] ~keepdim:true) in
  Tensor.(mean (neg (log (numerator / denominator))))

let state_dict t = Nn.state_dict t.lbl
let load_state_dict t state_dict = Nn.load_state_dict t.lbl state_dict