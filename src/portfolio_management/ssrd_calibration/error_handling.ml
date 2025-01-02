exception CalibrationError of string

let handle_error f =
  try f ()
  with
  | CalibrationError msg ->
      Printf.eprintf "Calibration Error: %s\n" msg;
      exit 1
  | Sys_error msg ->
      Printf.eprintf "System Error: %s\n" msg;
      exit 1
  | _ ->
      Printf.eprintf "Unknown Error occurred\n";
      exit 1