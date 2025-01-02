open Torch

type t = {
  layers: Layer.t list;
}

let create input_dim hidden_dims output_dim ~device =
  let dims = input_dim :: hidden_dims @ [output_dim] in
  let layers =
    List.map2 (fun d1 d2 -> Layer.linear d1 d2 ~device) dims (List.tl dims)
    |> List.mapi (fun i l ->
        if i = List.length dims - 2 then l  (* No activation on last layer *)
        else Layer.of_fn (fun x -> Tensor.(relu (Layer.forward l x))))
  in
  { layers }

let forward t x =
  List.fold_left (fun acc layer -> Layer.forward layer acc) x t.layers

let parameters t =
  List.concat_map Layer.parameters t.layers