open Torch
open Factor_construction
open Factor_model
open Data_loader

let compare_models configs stock_data =
  List.map (fun config ->
    let factors, returns = prepare_factors stock_data config.sort_method in
    let model = Factor_model.create config.learning_rate in
    let trained_model = Factor_model.train model factors returns config.num_epochs config.batch_size in
    let r_squared = Factor_model.calculate_r_squared trained_model factors returns in
    let weights = Factor_model.get_weights trained_model in
    (config.name, r_squared, weights)
  ) configs

let run_analysis stock_data batch_size num_bootstraps k_folds =
  let configs = [
    { name = "2x3 Sort"; sort_method = TwoByThree; learning_rate = 0.01; num_epochs = 1000; batch_size = batch_size };
    { name = "2x2 Sort"; sort_method = TwoByTwo; learning_rate = 0.01; num_epochs = 1000; batch_size = batch_size };
    { name = "2x2x2x2 Sort"; sort_method = TwoByTwoByTwoByTwo; learning_rate = 0.01; num_epochs = 1000; batch_size = batch_size };
  ] in
  
  let results = compare_models configs stock_data in
  
  List.iter (fun (name, r_squared, weights) ->
    Printf.printf "Model: %s\n" name;
    Printf.printf "R-squared: %f\n" r_squared;
    Printf.printf "Weights: [";
    Array.iter (fun w -> Printf.printf "%f; " w) weights;
    Printf.printf "]\n\n"
  ) results;

  let factors, returns = prepare_factors stock_data TwoByThree in
  let bootstrap_results = Model_validation.bootstrap_validation (Factor_model.create 0.01) factors returns num_bootstraps in
  Printf.printf "Bootstrap Results:\n";
  Printf.printf "Mean R-squared: %f\n" (List.fold_left (+.) 0. bootstrap_results /. float_of_int num_bootstraps);
  Printf.printf "Std Dev R-squared: %f\n" (Statistical_tests.calculate_std_dev bootstrap_results);

  let cv_results = Model_validation.k_fold_cross_validation (Factor_model.create 0.01) factors returns k_folds in
  Printf.printf "Cross-Validation Results:\n";
  Printf.printf "Mean R-squared: %f\n" (List.fold_left (+.) 0. cv_results /. float_of_int k_folds);
  Printf.printf "Std Dev R-squared: %f\n" (Statistical_tests.calculate_std_dev cv_results)