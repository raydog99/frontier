open Torch
open Pathwise_attention
open Geometric_attention
open Mlp

type t = {
  pathwise_attention: Pathwise_attention.t;
  encoder: Mlp.t;
  transition: Mlp.t;
  emission: Mlp.t;
  geometric_attention: Geometric_attention.t;
}

let create ~n_ref ~n_sim ~d_y ~n_pos ~n_time ~hidden_dims ~n_0 ~d_x ~device =
  let pathwise_attention = Pathwise_attention.create ~n_ref ~n_sim ~d_y ~n_pos ~n_time ~device in
  let encoder = Mlp.create (n_sim * d_y) hidden_dims n_0 ~device in
  let transition = Mlp.create d_x hidden_dims d_x ~device in
  let emission = Mlp.create d_x hidden_dims d_y ~device in
  let geometric_attention = Geometric_attention.create n_0 d_x ~device in
  { pathwise_attention; encoder; transition; emission; geometric_attention }

let forward t y =
  let encoded = Pathwise_attention.forward t.pathwise_attention y in
  let encoded = Mlp.forward t.encoder encoded in
  let mean, cov = Geometric_attention.forward t.geometric_attention encoded in
  let predicted_state = Mlp.forward t.transition mean in
  let predicted_obs = Mlp.forward t.emission predicted_state in
  (predicted_state, predicted_obs)

let parameters t =
  Pathwise_attention.parameters t.pathwise_attention @
  Mlp.parameters t.encoder @
  Mlp.parameters t.transition @
  Mlp.parameters t.emission @
  Geometric_attention.parameters t.geometric_attention

let save t ~filename =
  let state_dict = List.map (fun p -> (Tensor.name p, p)) (parameters t) in
  Serialize.save_multi ~named_tensors:state_dict ~filename

let load t ~filename =
  let named_tensors = Serialize.load_multi ~filename in
  List.iter (fun (name, tensor) ->
    match List.find_opt (fun p -> Tensor.name p = name) (parameters t) with
    | Some p -> Tensor.copy_ p tensor
    | None -> Printf.printf "Warning: tensor %s not found in model\n" name
  ) named_tensors