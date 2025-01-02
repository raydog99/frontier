open Torch
open Yojson.Basic.Util
open Sqlite3
open Factor_construction

type input_format = CSV | JSON | SQLite

let country_of_string = function
  | "US" -> US
  | "Japan" -> Japan
  | "UK" -> UK
  | "Canada" -> Canada
  | "France" -> France
  | "Germany" -> Germany
  | _ -> failwith "Unknown country"

let industry_of_string = function
  | "Technology" -> Technology
  | "Healthcare" -> Healthcare
  | "Finance" -> Finance
  | "ConsumerGoods" -> ConsumerGoods
  | "Energy" -> Energy
  | _ -> failwith "Unknown industry"

let load_csv filename =
  let ic = open_in filename in
  let headers = input_line ic |> String.split_on_char ',' in
  let data = ref [] in
  try
    while true do
      let line = input_line ic in
      let values = String.split_on_char ',' line in
      let stock = {
        date = int_of_string (List.nth values 0);
        country = country_of_string (List.nth values 1);
        industry = industry_of_string (List.nth values 2);
        size = float_of_string (List.nth values 3);
        bm = float_of_string (List.nth values 4);
        op = float_of_string (List.nth values 5);
        inv = float_of_string (List.nth values 6);
        returns = float_of_string (List.nth values 7);
        exchange = List.nth values 8;
      } in
      data := stock :: !data
    done;
    !data
  with End_of_file ->
    close_in ic;
    List.rev !data

let load_json filename =
  let json = Yojson.Basic.from_file filename in
  json |> to_list |> List.map (fun stock ->
    {
      date = stock |> member "Date" |> to_int;
      country = stock |> member "Country" |> to_string |> country_of_string;
      industry = stock |> member "Industry" |> to_string |> industry_of_string;
      size = stock |> member "Size" |> to_float;
      bm = stock |> member "B/M" |> to_float;
      op = stock |> member "OP" |> to_float;
      inv = stock |> member "Inv" |> to_float;
      returns = stock |> member "Returns" |> to_float;
      exchange = stock |> member "Exchange" |> to_string;
    }
  )

let load_sqlite filename =
  let db = db_open filename in
  let sql = "SELECT * FROM stocks" in
  let stmt = prepare db sql in
  let stocks = ref [] in
  while step stmt = Rc.ROW do
    stocks := {
      date = column_int stmt 0;
      country = column_text stmt 1 |> country_of_string;
      industry = column_text stmt 2 |> industry_of_string;
      size = column_float stmt 3;
      bm = column_float stmt 4;
      op = column_float stmt 5;
      inv = column_float stmt 6;
      returns = column_float stmt 7;
      exchange = column_text stmt 8;
    } :: !stocks
  done;
  finalize stmt |> ignore;
  db_close db |> ignore;
  List.rev !stocks

let load_data filename format =
  match format with
  | CSV -> load_csv filename
  | JSON -> load_json filename
  | SQLite -> load_sqlite filename

let prepare_factors stock_data sort_method =
  let factors = construct_factors stock_data sort_method in
  let returns = List.map (fun s -> s.returns) stock_data in
  let factors_tensor = factors_to_tensor [fst factors; snd factors; snd (snd factors); snd (snd (snd factors))] in
  let returns_tensor = Tensor.of_float1 (Array.of_list returns) in
  (factors_tensor, returns_tensor)