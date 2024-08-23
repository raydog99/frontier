type timing = {
  name: string;
  start_time: float;
  end_time: float;
  children: timing list;
}

type t = {
  mutable current: timing option;
  mutable stack: timing list;
  mutable root: timing option;
}

let create () = { current = None; stack = []; root = None }

let start profiler name =
  let timing = { name; start_time = Unix.gettimeofday (); end_time = 0.; children = [] } in
  match profiler.current with
  | None -> profiler.root <- Some timing
  | Some parent ->
      profiler.stack <- parent :: profiler.stack;
  profiler.current <- Some timing

let stop profiler =
  match profiler.current with
  | None -> ()
  | Some timing ->
      let ended_timing = { timing with end_time = Unix.gettimeofday () } in
      match profiler.stack with
      | [] -> profiler.root <- Some ended_timing
      | parent :: rest ->
          profiler.current <- Some { parent with children = ended_timing :: parent.children };
          profiler.stack <- rest

let rec print_timing indent timing =
  Printf.printf "%s%s: %.6f seconds\n" indent timing.name (timing.end_time -. timing.start_time);
  List.iter (print_timing (indent ^ "  ")) (List.rev timing.children)

let print profiler =
  match profiler.root with
  | None -> Printf.printf "No profiling data available.\n"
  | Some root -> print_timing "" root