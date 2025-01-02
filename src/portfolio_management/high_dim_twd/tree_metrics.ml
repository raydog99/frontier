let shortest_path_distance tree node1 node2 =
  let rec find_path node target visited path =
    if node = target then Some (List.rev path)
    else if List.mem node visited then None
    else
      let neighbors = Array.to_list (
        Array.mapi (fun i (src, dst) -> 
          if src = node then Some dst 
          else if dst = node then Some src 
          else None) tree.edges)
        |> List.filter_map (fun x -> x)
      in
      let rec try_paths = function
        | [] -> None
        | next :: rest ->
            match find_path next target (node :: visited) (next :: path) with
            | Some p -> Some p
            | None -> try_paths rest
      in
      try_paths neighbors
  in
  match find_path node1 node2 [] [node1] with
  | None -> infinity
  | Some path ->
      let total = ref 0. in
      List.iter2 (fun n1 n2 ->
        let edge_idx = Array.find_index (fun (src, dst) -> 
          (src = n1 && dst = n2) || (src = n2 && dst = n1)) tree.edges in
        total := !total +. tree.weights.(edge_idx)
      ) (List.init (List.length path - 1) (fun i -> List.nth path i))
         (List.init (List.length path - 1) (fun i -> List.nth path (i + 1)));
      !total