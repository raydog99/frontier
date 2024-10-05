type error_metrics = {
  rmse: float;
  mae: float;
  mape: float;
}

let calculate_rmse market_data model_data =
  let squared_errors = List.map2 (fun (_, m) (_, d) -> (m -. d) ** 2.0) market_data model_data in
  sqrt (List.fold_left (+.) 0.0 squared_errors /. float (List.length squared_errors))

let calculate_mae market_data model_data =
  let absolute_errors = List.map2 (fun (_, m) (_, d) -> abs_float (m -. d)) market_data model_data in
  List.fold_left (+.) 0.0 absolute_errors /. float (List.length absolute_errors)

let calculate_mape market_data model_data =
  let percentage_errors = List.map2 (fun (_, m) (_, d) -> abs_float ((m -. d) /. m) *. 100.0) market_data model_data in
  List.fold_left (+.) 0.0 percentage_errors /. float (List.length percentage_errors)

let calculate_error_metrics market_data model_data =
  {
    rmse = calculate_rmse market_data model_data;
    mae = calculate_mae market_data model_data;
    mape = calculate_mape market_data model_data;
  }