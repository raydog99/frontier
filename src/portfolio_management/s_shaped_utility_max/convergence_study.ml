open Torch

let convergence_study train_fn evaluate_fn params data_loader iterations_list =
  List.map (fun iterations ->
    let start_time = Unix.gettimeofday () in
    train_fn params data_loader iterations;
    let end_time = Unix.gettimeofday () in
    let training_time = end_time -. start_time in
    let error = Error_analysis.mean_squared_error (evaluate_fn params) (Data_handling.get_test_data data_loader) in
    (iterations, Tensor.float_value error, training_time)
  ) iterations_list

let print_convergence_results results =
  Printf.printf "Convergence study results:\n";
  Printf.printf "Iterations | MSE | Training Time (s)\n";
  List.iter (fun (iterations, mse, time) ->
    Printf.printf "%10d | %.6f | %.2f\n" iterations mse time
  ) results