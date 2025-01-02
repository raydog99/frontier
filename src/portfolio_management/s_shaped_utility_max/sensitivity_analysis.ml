open Torch

let parameter_sensitivity model params evaluate_fn parameter_name range num_points =
  let original_value = Hashtbl.find params parameter_name in
  let values = List.init num_points (fun i ->
    let t = float_of_int i /. float_of_int (num_points - 1) in
    let (low, high) = range in
    low +. t *. (high -. low)
  ) in
  let results = List.map (fun value ->
    Hashtbl.replace params parameter_name value;
    let result = evaluate_fn model params in
    Hashtbl.replace params parameter_name original_value;
    (value, result)
  ) values in
  results

let print_sensitivity_results parameter_name results =
  Printf.printf "Sensitivity analysis for %s:\n" parameter_name;
  List.iter (fun (value, result) ->
    Printf.printf "  %s = %.4f: %.4f\n" parameter_name value (Tensor.float_value result)
  ) results