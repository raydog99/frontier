type hyperparameters = {
  alpha: float;
  max_iterations: int;
  tolerance: float;
  rel_tolerance: float;
  stagnation_window: int;
  adaptive_step: bool;
}

let random_search n_iter obj_fn param_ranges =
  let sample_param range =
    let (min_val, max_val) = range in
    min_val +. (max_val -. min_val) *. Random.float 1.0
  in
  let sample_params () =
    { alpha = sample_param param_ranges.alpha;
      max_iterations = int_of_float (sample_param param_ranges.max_iterations);
      tolerance = sample_param param_ranges.tolerance;
      rel_tolerance = sample_param param_ranges.rel_tolerance;
      stagnation_window = int_of_float (sample_param param_ranges.stagnation_window);
      adaptive_step = Random.bool () }
  in
  let rec search i best_params best_score =
    if i >= n_iter then (best_params, best_score)
    else
      let params = sample_params () in
      let score = obj_fn params in
      if score < best_score then
        search (i + 1) params score
      else
        search (i + 1) best_params best_score
  in
  search 0 (sample_params ()) Float.infinity

let grid_search obj_fn param_grid =
  let rec cartesian_product = function
    | [] -> [[]]
    | h::t ->
        let rest = cartesian_product t in
        List.concat (List.map (fun x -> List.map (fun y -> x::y) rest) h)
  in
  let param_combinations = cartesian_product [
    param_grid.alpha;
    List.map float_of_int param_grid.max_iterations;
    param_grid.tolerance;
    param_grid.rel_tolerance;
    List.map float_of_int param_grid.stagnation_window;
    [true; false]
  ] in
  let evaluate_params params =
    let [alpha; max_iterations; tolerance; rel_tolerance; stagnation_window; adaptive_step] = params in
    let hp = {
      alpha = alpha;
      max_iterations = int_of_float max_iterations;
      tolerance = tolerance;
      rel_tolerance = rel_tolerance;
      stagnation_window = int_of_float stagnation_window;
      adaptive_step = adaptive_step = 1.0
    } in
    (hp, obj_fn hp)
  in
  List.fold_left (fun (best_params, best_score) params ->
    let (hp, score) = evaluate_params params in
    if score < best_score then (hp, score) else (best_params, best_score)
  ) (evaluate_params (List.hd param_combinations)) (List.tl param_combinations)