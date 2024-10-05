open Types

let read_market_data zcb_file cds_file =
  let read_file filename =
    let ic = open_in filename in
    let rec read_lines acc =
      try
        let line = input_line ic in
        let parts = String.split_on_char ',' line in
        let t = float_of_string (List.nth parts 0) in
        let value = float_of_string (List.nth parts 1) in
        read_lines ((t, value) :: acc)
      with End_of_file ->
        close_in ic;
        List.rev acc
    in
    read_lines []
  in
  {
    zcb_prices = read_file zcb_file;
    cds_spreads = read_file cds_file;
  }

let calculate_errors model_data market_data =
  let zcb_error = List.fold_left2 (fun acc (_, model_price) (_, market_price) ->
    acc +. (model_price -. market_price) ** 2.0
  ) 0.0 model_data market_data.zcb_prices in
  let cds_error = List.fold_left2 (fun acc (_, model_spread) (_, market_spread) ->
    acc +. (model_spread -. market_spread) ** 2.0
  ) 0.0 model_data market_data.cds_spreads in
  (sqrt (zcb_error /. float (List.length market_data.zcb_prices)),
   sqrt (cds_error /. float (List.length market_data.cds_spreads)))