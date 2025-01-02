open Torch

type noise_dist = 
  | Normal of float * float
  | Uniform of float * float
  | Custom of (unit -> float)

type structural_fn = {
  fn: Tensor.t -> Tensor.t -> Tensor.t;
  noise: noise_dist;
}

type t = {
  variables: string array;
  domains: int array;
  graph: (int * int) list;
  functions: structural_fn array;
}

let sample_noise = function
  | Normal (mu, sigma) -> Tensor.normal ~mean:mu ~std:sigma [1]
  | Uniform (min_val, max_val) -> 
      Tensor.(add (mul_scalar (rand [1]) (max_val -. min_val)) (float min_val))
  | Custom f -> Tensor.of_float1 [|f ()|]

let topological_sort scm =
  let n = Array.length scm.variables in
  let adj = Array.make n [] in
  let in_degree = Array.make n 0 in
  
  List.iter (fun (p, c) ->
    adj.(p) <- c :: adj.(p);
    in_degree.(c) <- in_degree.(c) + 1
  ) scm.graph;
  
  let q = Queue.create () in
  for i = 0 to n - 1 do
    if in_degree.(i) = 0 then Queue.add i q
  done;
  
  let rec process acc =
    if Queue.is_empty q then List.rev acc
    else
      let v = Queue.take q in
      let acc' = v :: acc in
      List.iter (fun u ->
        in_degree.(u) <- in_degree.(u) - 1;
        if in_degree.(u) = 0 then Queue.add u q
      ) adj.(v);
      process acc'
  in
  process []

let sample scm n =
  let order = topological_sort scm in
  let samples = Array.make (Array.length scm.variables) (Tensor.zeros [n]) in
  
  List.iter (fun v ->
    let parent_vals = List.filter_map (fun (p, c) ->
      if c = v then Some samples.(p) else None
    ) scm.graph in
    
    let noise = sample_noise scm.functions.(v).noise in
    
    let parents = match parent_vals with
      | [] -> Tensor.zeros [n]
      | [x] -> x
      | xs -> Tensor.cat xs ~dim:1 in
    samples.(v) <- scm.functions.(v).fn parents noise
  ) order;
  
  Tensor.cat (Array.to_list samples) ~dim:1

let apply_intervention scm intervention =
  let new_functions = Array.copy scm.functions in
  Array.iteri (fun i _ ->
    if Array.mem i intervention.variables then
      let idx = Array.find_index ((=) i) intervention.variables |> Option.get in
      new_functions.(i) <- {
        fn = (fun _ _ -> intervention.func (Tensor.of_float1 [|intervention.values.(idx)|]));
        noise = Normal(0.0, 0.0)
      }
  ) new_functions;
  {scm with functions = new_functions}