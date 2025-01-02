type t = {
  mutable prev_estimates: float list;
  window_size: int;
  tolerance: float;
}

let create ~window_size ~tolerance =
  { prev_estimates = []; window_size; tolerance }

let should_stop criterion current_estimate =
  criterion.prev_estimates <- current_estimate :: criterion.prev_estimates;
  if List.length criterion.prev_estimates > criterion.window_size then
    criterion.prev_estimates <- List.take criterion.window_size criterion.prev_estimates;
  
  if List.length criterion.prev_estimates < criterion.window_size then
    false
  else
    let mean = List.fold_left (+.) 0. criterion.prev_estimates /. float_of_int criterion.window_size in
    let std_dev = sqrt (List.fold_left (fun acc x -> acc +. (x -. mean) ** 2.) 0. criterion.prev_estimates
                        /. float_of_int criterion.window_size) in
    std_dev < criterion.tolerance