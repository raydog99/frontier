open Types

let rec compute_maximal_chains interventions =
  let can_extend chain i =
    match List.rev chain with
    | [] -> true
    | last :: _ -> intervention_leq last i
  in
  
  let rec build_chains acc current remaining =
    match remaining with
    | [] -> if current = [] then acc else current :: acc
    | i :: rest ->
        if can_extend current i then
          let with_i = build_chains acc (i :: current) rest in
          let without_i = build_chains with_i current rest in
          without_i
        else
          build_chains acc current rest
  in
  
  let chains = build_chains [] [] interventions in
  
  List.filter (fun c1 ->
    not (List.exists (fun c2 ->
      c1 <> c2 && List.length c1 < List.length c2 &&
      List.for_all2 intervention_leq c1 (List.take c2 (List.length c1))
    ) chains)
  ) chains

let check_comparability i1 i2 chains =
  let containing_chain = List.find_opt (fun chain ->
    let i1_idx = List.find_index ((=) i1) chain in
    let i2_idx = List.find_index ((=) i2) chain in
    match i1_idx, i2_idx with
    | Some idx1, Some idx2 -> abs(idx1 - idx2) = 1
    | _ -> false
  ) chains in
  match containing_chain with
  | Some chain -> Comparable chain
  | None -> NotComparable