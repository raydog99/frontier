open Torch

type scenario = {
  probability: float;
  volatility_multiplier: float;
  drift: float;
  permanent_impact_multiplier: float;
  temporary_impact_multiplier: float;
}

type t = {
  shift_time: float;
  scenarios: scenario list;
}

let create shift_time scenarios = { shift_time; scenarios }

let apply t price_path =
  let n = Tensor.shape price_path |> List.hd in
  let shift_index = int_of_float (t.shift_time *. float_of_int n) in
  let pre_shift = Tensor.narrow price_path ~dim:0 ~start:0 ~length:shift_index in
  let post_shift = Tensor.narrow price_path ~dim:0 ~start:shift_index ~length:(n - shift_index) in
  
  let shifted_paths = List.map (fun scenario ->
    let shifted_post = Tensor.(
      post_shift * (f scenario.volatility_multiplier) +
      (f scenario.drift * arange ~start:0. ~end_:(float_of_int (n - shift_index)) ~options:(T Float))
    ) in
    (scenario.probability, Tensor.cat [pre_shift; shifted_post] ~dim:0)
  ) t.scenarios in
  
  let probabilities = List.map (fun (prob, _) -> prob) shifted_paths |> Tensor.of_float1 in
  let paths = List.map (fun (_, path) -> path) shifted_paths |> Tensor.stack ~dim:0 in
  
  Tensor.sum (Tensor.mul paths (Tensor.unsqueeze probabilities ~dim:1)) ~dim:0

let get_expected_parameters t initial_volatility initial_permanent_impact initial_temporary_impact =
  List.fold_left (fun (vol, perm, temp) scenario ->
    let vol' = vol +. scenario.probability *. scenario.volatility_multiplier *. initial_volatility in
    let perm' = perm +. scenario.probability *. scenario.permanent_impact_multiplier *. initial_permanent_impact in
    let temp' = temp +. scenario.probability *. scenario.temporary_impact_multiplier *. initial_temporary_impact in
    (vol', perm', temp')
  ) (0., 0., 0.) t.scenarios