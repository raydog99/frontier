open Types

let create_omega_map base_scm abstract_scm var_mapping =
  fun base_int ->
    (* Map intervention variables using var_mapping *)
    let abstract_vars = List.filter_map (fun (base_idx, abstract_idx) ->
      if Array.mem base_idx base_int.variables then Some abstract_idx
      else None
    ) var_mapping |> Array.of_list in
    
    (* Map intervention values *)
    let abstract_vals = Array.map (fun abstract_idx ->
      let base_idx = List.find_opt (fun (b, a) -> a = abstract_idx) var_mapping in
      match base_idx with
      | Some (b, _) ->
          let idx = Array.find_index ((=) b) base_int.variables |> Option.get in
          base_int.values.(idx)
      | None -> 0.0  (* Default value if no mapping exists *)
    ) abstract_vars in
    
    (* Create abstract intervention function *)
    let abstract_func = fun x -> 
      Tensor.of_float1 abstract_vals in
    
    {variables = abstract_vars; values = abstract_vals; func = abstract_func}

let is_order_preserving omega i1 i2 =
  if intervention_leq i1 i2 then
    let omega_i1 = omega i1 in
    let omega_i2 = omega i2 in
    intervention_leq omega_i1 omega_i2
  else true