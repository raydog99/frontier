exception InvalidData of string
exception ComputationError of string
exception StrategyError of string

let handle_exn f =
  try Ok (f ())
  with
  | InvalidData msg -> Error ("Invalid data: " ^ msg)
  | ComputationError msg -> Error ("Computation error: " ^ msg)
  | StrategyError msg -> Error ("Strategy error: " ^ msg)
  | Failure msg -> Error ("Failure: " ^ msg)
  | _ -> Error "Unknown error occurred"