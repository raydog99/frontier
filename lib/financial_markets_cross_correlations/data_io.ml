open Types
open Yojson.Basic.Util

exception InvalidInputData of string

let read_market_data filename =
  let lines = ref [] in
  let chan = open_in filename in
  try
    while true do
      lines := input_line chan :: !lines
    done;
    assert false 
  with End_of_file ->
    close_in chan;
    let data = List.rev !lines in
    match data with
    | header :: rest ->
      let markets = String.split_on_char ',' header |> List.tl in
      let prices = List.map (fun line ->
        let values = String.split_on_char ',' line |> List.tl in
        List.map float_of_string values
      ) rest in
      let transposed_prices = List.transpose prices in
      Array.of_list (List.mapi (fun i market ->
        { name = market; prices = Array.of_list (List.nth transposed_prices i) }
      ) markets)
    | [] -> raise (InvalidInputData "Empty input file")

let save_results (summary: analysis_summary) filename =
  let json = `Assoc [
    "dccc_results", `List (Array.to_list (Array.map (fun r ->
      `Assoc [
        "short_scale", `Int r.short_scale;
        "long_scale", `Int r.long_scale;
        "dccc", `Float r.dccc;
        "timestamp", `Float r.timestamp;
      ]
    ) summary.dccc_results));
    "regime_results", `List (Array.to_list (Array.map (fun r ->
      `Assoc [
        "timestamp", `Float r.timestamp;
        "regime", `String (match r.regime with Bull -> "Bull" | Bear -> "Bear" | Neutral -> "Neutral");
      ]
    ) summary.regime_results));
    "influential_markets", `List (List.map (fun (name, centrality) ->
      `Assoc [
        "name", `String name;
        "centrality", `Float centrality;
      ]
    ) summary.influential_markets);
    "market_efficiencies", `List (Array.to_list (Array.map (fun me ->
      `Assoc [
        "name", `String me.name;
        "hurst_exponent", `Float me.hurst_exponent;
      ]
    ) summary.market_efficiencies));
    "volatility_clusters", `List (Array.to_list (Array.map (fun vc ->
      `Assoc [
        "start_timestamp", `Float vc.start_timestamp;
        "end_timestamp", `Float vc.end_timestamp;
        "intensity", `Float vc.intensity;
      ]
    ) summary.volatility_clusters));
  ] in
  Yojson.Basic.to_file filename json