open Torch
open Graph

type t = {
  dim: int;
  weights: Tensor.t;
  intervention_weights: Tensor.t;
  noise_mean: Tensor.t;
  max_weight: float;
  max_noise: float;
}

let create ?(max_weight=1.0) ?(max_noise=1.0) dim =
  {
    dim;
    weights = Tensor.zeros [dim; dim];
    intervention_weights = Tensor.zeros [dim; dim];
    noise_mean = Tensor.zeros [dim];
    max_weight;
    max_noise;
  }

let simulate model intervention x =
  let open Tensor in
  (* For nodes in intervention, use intervention weights *)
  let weights = model.weights in
  let int_weights = model.intervention_weights in
  
  NodeSet.fold (fun node acc ->
    let row = narrow weights ~dim:0 ~start:node ~length:1 in
    let int_row = narrow int_weights ~dim:0 ~start:node ~length:1 in
    copy_ acc ~src:int_row ~start0:node ~length0:1
  ) intervention weights
  |> fun w -> mm w x + model.noise_mean

let estimate_means observations intervention =
  let open Tensor in
  (* Calculate μ̂i,∅ and μ̂i,{j} *)
  let null_obs = List.filter (fun (_, i) -> NodeSet.is_empty i) observations in
  let int_obs = List.filter (fun (_, i) -> not (NodeSet.is_empty i)) observations in
  
  let null_mean = 
    List.map fst null_obs |> Stats.sample_mean
  in
  
  let int_means = List.fold_left (fun acc (obs, int) ->
    let node = NodeSet.choose int in
    NodeMap.add node (Stats.mean obs) acc  
  ) NodeMap.empty int_obs in
  
  (null_mean, int_means)

let check_weight_constraint model =
  let open Tensor in
  let check_matrix m = 
    max m |> float_value <= model.max_weight &&
    min m |> float_value >= -.model.max_weight
  in
  check_matrix model.weights && check_matrix model.intervention_weights

let check_intervention_regularity model eta observations =
  let dim = model.dim in
  let check_node i =
    let null_mean, int_means = estimate_means observations (NodeSet.singleton i) in
    NodeMap.for_all (fun j mean ->
      if j > i then
        abs_float (Tensor.float_value (mean - null_mean)) > eta
      else true
    ) int_means
  in
  List.init dim (fun i -> i)
  |> List.for_all check_node