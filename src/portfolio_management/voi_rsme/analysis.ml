open Torch

let train_model data_type predictors response =
  match data_type with
  | `LinearRegression -> Model.train_linear_regression predictors response
  | `PLS components -> Model.train_pls predictors response components
  | `NeuralNetwork -> Model.train_neural_network predictors response

let evaluate_model model predictors response =
  let predictions = Model.predict model predictors in
  let rmse = Metrics.rmse predictions response in
  let correlation = Metrics.correlation predictions response in
  let mrr = Metrics.mean_rate_of_return predictions response in
  (rmse, correlation, mrr)

let analyze_lags_and_symbols dataset max_lags max_symbols =
  let results = ref [] in
  for num_lags = 2 to max_lags do
    for num_symbols = 1 to max_symbols do
      try
        let windows = Dataset.create_rolling_windows dataset 100 25 in
        let model_results = List.map (fun (train_data, test_data) ->
          let train_predictors = Dataset.create_predictors train_data num_lags num_symbols in
          let train_response = Dataset.create_response train_data num_lags in
          let test_predictors = Dataset.create_predictors test_data num_lags num_symbols in
          let test_response = Dataset.create_response test_data num_lags in
          
          let information = Voi.calculate_mutual_information train_predictors train_response in
          
          let models = [
            ("Linear Regression", train_model `LinearRegression train_predictors train_response);
            ("PLS", train_model (`PLS 3) train_predictors train_response);
            ("Neural Network", train_model `NeuralNetwork train_predictors train_response);
          ] in
          
          List.map (fun (name, model) ->
            let _, test_corr, test_mrr = evaluate_model model test_predictors test_response in
            (num_lags, num_symbols, name, information, test_corr, test_mrr)
          ) models
        ) windows |> List.concat in
        
        results := model_results @ !results
      with e ->
        Printf.eprintf "Error in analysis for lags=%d, symbols=%d: %s\n" num_lags num_symbols (Printexc.to_string e)
    done
  done;
  !results