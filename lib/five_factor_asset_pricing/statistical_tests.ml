let calculate_mean data =
  List.fold_left (+.) 0. data /. float_of_int (List.length data)

let calculate_std_dev data =
  let mean = calculate_mean data in
  let variance = List.fold_left (fun acc x -> acc +. (x -. mean) ** 2.) 0. data /. float_of_int (List.length data) in
  sqrt variance

let calculate_t_statistic mean std_dev n =
  mean /. (std_dev /. sqrt (float_of_int n))

let calculate_p_value t_stat df =
  let normal_cdf x =
    let t = 1. /. (1. +. 0.2316419 *. abs_float x) in
    let d = 0.3989423 *. exp (-.(x*.x)/.2.) in
    let p = d *. t *. (0.3193815 +. t *. (-0.3565638 +. t *. (1.781478 +. t *. (-1.821256 +. t *. 1.330274)))) in
    if x > 0. then 1. -. p else p
  in
  2. *. (1. -. normal_cdf (abs_float t_stat))

let perform_t_test data =
  let mean = calculate_mean data in
  let std_dev = calculate_std_dev data in
  let n = List.length data in
  let t_stat = calculate_t_statistic mean std_dev n in
  let p_value = calculate_p_value t_stat (float_of_int (n - 1)) in
  (mean, std_dev, t_stat, p_value)