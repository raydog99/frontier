type t = {
  mutable data: (string * (string * float) list) list;
}

let create () = { data = [] }

let load_from_csv t filename =
  let ic = open_in filename in
  let headers = input_line ic |> String.split_on_char ',' in
  try
    while true do
      let line = input_line ic in
      let values = line |> String.split_on_char ',' in
      let symbol = List.hd values in
      let fundamental_data = List.combine (List.tl headers) (List.tl values |> List.map float_of_string) in
      t.data <- (symbol, fundamental_data) :: t.data
    done
  with End_of_file ->
    close_in ic

let get_fundamental t symbol key =
  match List.assoc_opt symbol t.data with
  | Some fundamentals -> List.assoc_opt key fundamentals
  | None -> None

let update_fundamental t symbol key value =
  let updated_fundamentals =
    match List.assoc_opt symbol t.data with
    | Some fundamentals -> (key, value) :: List.remove_assoc key fundamentals
    | None -> [(key, value)]
  in
  t.data <- (symbol, updated_fundamentals) :: List.remove_assoc symbol t.data