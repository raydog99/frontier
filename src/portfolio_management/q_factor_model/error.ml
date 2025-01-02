exception DataError of string
exception ModelError of string
exception StatisticalError of string
exception VisualizationError of string

let handle_error f =
  try f ()
  with
  | DataError msg -> Printf.eprintf "Data Error: %s\n" msg
  | ModelError msg -> Printf.eprintf "Model Error: %s\n" msg
  | StatisticalError msg -> Printf.eprintf "Statistical Error: %s\n" msg
  | VisualizationError msg -> Printf.eprintf "Visualization Error: %s\n" msg
  | e -> Printf.eprintf "Unexpected error: %s\n" (Printexc.to_string e)