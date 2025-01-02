type rate_function = Portfolio.t -> float

let calculate_rate_function portfolios market_seq t =
  let w_star = List.fold_left (fun acc p ->
    max acc (log (Portfolio.relative_value p market_seq t) /. float_of_int t)
  ) neg_infinity portfolios in
  fun p -> w_star -. (log (Portfolio.relative_value p market_seq t) /. float_of_int t)

let verify_ldp distribution rate_fn epsilon =
  let open WealthDistribution in
  let dist = get_distribution distribution in
  List.for_all (fun (p, w) ->
    abs_float (log w /. float_of_int (Portfolio.size p) +. rate_fn p) < epsilon
  ) dist

let asymptotic_equipartition portfolios market_seq t epsilon =
  let rate_fn = calculate_rate_function portfolios market_seq t in
  let dist = WealthDistribution.create (List.map (fun p -> FunctionallyGeneratedPortfolio.create "Custom" (fun _ -> Portfolio.get_weights p) (fun _ -> 0.0)) portfolios)
             |> fun d -> WealthDistribution.update d market_seq t in
  verify_ldp dist rate_fn epsilon

let varadhan_laplace_principle f portfolios market_seq t =
  let rate_fn = calculate_rate_function portfolios market_seq t in
  let max_value = ref neg_infinity in
  List.iter (fun p ->
    let value = f p -. rate_fn p in
    if value > !max_value then max_value := value
  ) portfolios;
  !max_value

let large_deviation_upper_bound portfolios rate_fn =
  let inf_rate = List.fold_left (fun acc p ->
    min acc (rate_fn p)
  ) infinity portfolios in
  -. inf_rate

let large_deviation_lower_bound portfolios rate_fn =
  let inf_rate = List.fold_left (fun acc p ->
    min acc (rate_fn p)
  ) infinity portfolios in
  -. inf_rate

let gartner_ellis_theorem cumulative_gen_fn t epsilon max_iterations =
  let lambda = ref 0.0 in
  let delta_lambda = 0.01 in
  let rec find_root iterations =
    if iterations = 0 then None
    else
      let value = cumulative_gen_fn !lambda t in
      if abs_float value < epsilon then Some !lambda
      else begin
        lambda := !lambda +. delta_lambda *. value;
        find_root (iterations - 1)
      end
  in
  find_root max_iterations

let scaled_cumulant_generating_function portfolios market_seq t =
  fun lambda ->
    let n = float_of_int (List.length portfolios) in
    (1.0 /. float_of_int t) *. log (
      List.fold_left (fun acc p ->
        acc +. (1.0 /. n) *. exp (lambda *. float_of_int t *. Portfolio.log_relative_value p market_seq t)
      ) 0.0 portfolios
    )

let rate_function_from_scgf scgf t =
  fun x ->
    let f lambda = scgf lambda t -. lambda *. x in
    let (a, b) = (-100.0, 100.0) in
    let rec golden_section_search a b tol =
      let phi = (sqrt 5.0 +. 1.0) /. 2.0 in
      let c = b -. (b -. a) /. phi in
      let d = a +. (b -. a) /. phi in
      if abs_float (c -. d) < tol then (f c)
      else
        if f c < f d then golden_section_search a d (tol *. 0.99)
        else golden_section_search c b (tol *. 0.99)
    in
    golden_section_search a b 1e-6

let verify_ldp_with_gartner_ellis portfolios market_seq t epsilon =
  let scgf = scaled_cumulant_generating_function portfolios market_seq in
  let rate_fn = rate_function_from_scgf scgf t in
  let empirical_mean = List.fold_left (fun acc p ->
    acc +. Portfolio.log_relative_value p market_seq t
  ) 0.0 portfolios /. float_of_int (List.length portfolios) in
  abs_float (rate_fn empirical_mean) < epsilon