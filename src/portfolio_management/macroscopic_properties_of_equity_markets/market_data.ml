open Torch
open Error
open Logging

module CRSPData = struct
  type t = {
    market_caps : float array array;
    dates : string array;
  }

  let load_data file_path =
    Logging.info ("Loading data from " ^ file_path);
    match Error.handle_exn (fun () ->
      let market_caps = [| [| 1000.; 2000.; 3000. |]; [| 1100.; 2200.; 3300. |] |] in
      let dates = [| "2024-01-01"; "2024-01-02" |] in
      { market_caps; dates }
    ) with
    | Ok data -> data
    | Error msg ->
        Logging.error ("Failed to load data: " ^ msg);
        raise (InvalidData msg)

  let get_market_caps data = data.market_caps
  let get_dates data = data.dates

  let get_returns data =
    let n = Array.length data.market_caps in
    let m = Array.length data.market_caps.(0) in
    Array.init (n - 1) (fun i ->
      Array.init m (fun j ->
        data.market_caps.(i+1).(j) /. data.market_caps.(i).(j) -. 1.
      )
    )

  let get_log_returns data =
    let returns = get_returns data in
    Array.map (Array.map log) returns

  let filter_by_date_range data start_date end_date =
    let start_index = Array.index_of (fun d -> d >= start_date) data.dates in
    let end_index = Array.index_of (fun d -> d > end_date) data.dates in
    let filtered_dates = Array.sub data.dates start_index (end_index - start_index) in
    let filtered_market_caps = Array.sub data.market_caps start_index (end_index - start_index) in
    { market_caps = filtered_market_caps; dates = filtered_dates }

  let filter_top_n_stocks data n =
    let m = Array.length data.market_caps.(0) in
    let top_n_indices = 
      Array.init m (fun i -> i)
      |> Array.sort (fun i j -> compare data.market_caps.(0).(j) data.market_caps.(0).(i))
      |> Array.sub 0 n
    in
    let filtered_market_caps = 
      Array.map (fun caps -> 
        Array.map (fun i -> caps.(i)) top_n_indices
      ) data.market_caps
    in
    { market_caps = filtered_market_caps; dates = data.dates }
end