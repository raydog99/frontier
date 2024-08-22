open Torch
open Logging
open Config

(** Financial indicator type *)
type indicator =
  | Y10  (** 10-Year Treasury Yield *)
  | CAPE (** Cyclically Adjusted Price/Earnings Ratio *)
  | NYF  (** New York Fed Economic Activity Index *)
  | MG   (** US Corporate Margins *)
  | Y02  (** 2-Year Treasury Yield *)
  | STP  (** Steepness of the Treasury Yield Curve *)
  | M2   (** Money Supply *)

type t = {
  target: Tensor.t;
  contexts: (indicator * Tensor.t) array;
  horizon: int;
}

let indicator_of_string = function
  | "Y10" -> Y10
  | "CAPE" -> CAPE
  | "NYF" -> NYF
  | "MG" -> MG
  | "Y02" -> Y02
  | "STP" -> STP
  | "M2" -> M2
  | s -> failwith (Printf.sprintf "Unknown indicator: %s" s)

let string_of_indicator = function
  | Y10 -> "Y10"
  | CAPE -> "CAPE"
  | NYF -> "NYF"
  | MG -> "MG"
  | Y02 -> "Y02"
  | STP -> "STP"
  | M2 -> "M2"

let load_csv filename =
  Logging.info (Printf.sprintf "Loading CSV file: %s" filename);
  let ic = open_in filename in
  let headers = input_line ic |> String.split_on_char ',' in
  let data = ref [] in
  try
    while true do
      let line = input_line ic in
      let values = line |> String.split_on_char ',' |> List.map float_of_string in
      data := values :: !data
    done;
    assert false
  with End_of_file ->
    close_in ic;
    Logging.info (Printf.sprintf "Loaded %d rows of data" (List.length !data));
    (headers, List.rev !data)

let load_data config =
  let headers, raw_data = load_csv config.Config.data_file in
  let target_index = List.find_index ((=) "S&P500") headers |> Option.get in
  let context_indices = List.filter_map (fun h ->
    match indicator_of_string h with
    | exception Failure _ -> None
    | ind -> Some (ind, List.find_index ((=) h) headers |> Option.get)
  ) headers in
  
  let data_tensor = Tensor.of_float2 (Array.of_list raw_data) in
  let target = Tensor.slice data_tensor ~dim:1 ~start:target_index ~end_:(target_index + 1) in
  let contexts = Array.of_list (List.map (fun (ind, idx) ->
    (ind, Tensor.slice data_tensor ~dim:1 ~start:idx ~end_:(idx + 1))
  ) context_indices) in
  
  Logging.info (Printf.sprintf "Loaded data with %d contexts" (Array.length contexts));
  { target; contexts; horizon = config.Config.horizon }

let prepare_input t index context_index =
  let target = Tensor.slice t.target ~dim:0 ~start:index ~end_:(index + t.horizon) in
  let _, context = t.contexts.(context_index) in
  let context_slice = Tensor.slice context ~dim:0 ~start:index ~end_:(index + t.horizon) in
  Tensor.cat [target; context_slice] ~dim:1

let shift_target t shifts =
  let shifted = Tensor.slice t.target ~dim:0 ~start:shifts ~end_:Tensor.shape_dim t.target 0 in
  let padding = Tensor.zeros [shifts; 1] in
  Tensor.cat [shifted; padding] ~dim:0