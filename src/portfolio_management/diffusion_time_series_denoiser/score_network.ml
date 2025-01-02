open Torch

type t = {
  model: Nn.t;
}

let create input_dim hidden_dim =
  let model = 
    Nn.sequential
      [
        Nn.linear input_dim hidden_dim;
        Nn.relu ();
        Nn.linear hidden_dim hidden_dim;
        Nn.relu ();
        Nn.linear hidden_dim input_dim;
      ]
  in
  { model }

let forward t x time_embed =
  let x_with_time = Tensor.cat [x; time_embed] ~dim:1 in
  Nn.Module.forward t.model x_with_time

let loss t sde x t =
  let z = Tensor.randn_like x in
  let mean, std = sde.sample x t z in
  let perturbed_x = Tensor.(mean + std * z) in
  let score = forward t perturbed_x (Tensor.float_vec [|t|]) in
  let target = Tensor.(neg z / float std) in
  Tensor.mse_loss score target ~reduction:Torch_core.Reduction.Mean

let to_serialize t =
  Serialize.to_string (Nn.Module.state_dict t.model)

let of_serialize input_dim hidden_dim s =
  let model = create input_dim hidden_dim in
  Nn.Module.load_state_dict model.model (Serialize.of_string s);
  model