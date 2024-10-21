let empirical_measure portfolios market_seq t =
  List.map (fun p ->
    Portfolio.log_relative_value p market_seq t /. float_of_int t
  ) portfolios

let true_measure portfolios market =
  List.map (fun p ->
    let weights = Portfolio.get_weights p in
    Array.fold_left2 (fun acc w m -> acc +. w *. log m) 0.0 weights (Market.get_weights market)
  ) portfolios

let supremum_norm emp_measure true_measure =
  List.fold_left2 (fun acc emp true_val ->
    max acc (abs_float (emp -. true_val))
  ) 0.0 emp_measure true_measure

let verify_gc_property portfolios market_seq t epsilon =
  let emp_measure = empirical_measure portfolios market_seq t in
  let true_measure = true_measure portfolios (List.nth market_seq t) in
  supremum_norm emp_measure true_measure < epsilon

let vc_dimension portfolios =
  let n = Portfolio.size (List.hd portfolios) in
  min (2 * n) (List.length portfolios)

let vapnik_chervonenkis_bound portfolios market_seq t epsilon =
  let d = vc_dimension portfolios in
  let bound = sqrt ((2.0 *. log (2.0 *. float_of_int d) +. log (4.0 /. epsilon)) /. float_of_int t) in
  verify_gc_property portfolios market_seq t bound

let rademacher_complexity portfolios market_seq t =
  let n = List.length portfolios in
  let sum_rademacher = ref 0.0 in
  for _ = 1 to 1000 do
    let rademacher = List.init n (fun _ -> if Random.bool () then 1.0 else -1.0) in
    let max_value = List.fold_left2 (fun acc sigma p ->
      max acc (sigma *. Portfolio.log_relative_value p market_seq t)
    ) neg_infinity rademacher portfolios in
    sum_rademacher := !sum_rademacher +. max_value
  done;
  !sum_rademacher /. (1000.0 *. float_of_int t)

let rademacher_bound portfolios market_seq t epsilon =
  let complexity = rademacher_complexity portfolios market_seq t in
  let bound = 2.0 *. complexity +. sqrt (2.0 *. log (1.0 /. epsilon) /. float_of_int t) in
  verify_gc_property portfolios market_seq t bound