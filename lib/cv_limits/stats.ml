open Base

let normal_quantile p =
  (* Approximation of the inverse normal CDF *)
  let a1 = -3.969683028665376e+01 in
  let a2 = 2.209460984245205e+02 in
  let a3 = -2.759285104469687e+02 in
  let a4 = 1.383577518672690e+02 in
  let a5 = -3.066479806614716e+01 in
  let a6 = 2.506628277459239e+00 in
  
  let b1 = -5.447609879822406e+01 in
  let b2 = 1.615858368580409e+02 in
  let b3 = -1.556989798598866e+02 in
  let b4 = 6.680131188771972e+01 in
  let b5 = -1.328068155288572e+01 in
  
  let c1 = -7.784894002430293e-03 in
  let c2 = -3.223964580411365e-01 in
  let c3 = -2.400758277161838e+00 in
  let c4 = -2.549732539343734e+00 in
  let c5 = 4.374664141464968e+00 in
  let c6 = 2.938163982698783e+00 in
  
  let d1 = 7.784695709041462e-03 in
  let d2 = 3.224671290700398e-01 in
  let d3 = 2.445134137142996e+00 in
  let d4 = 3.754408661907416e+00 in
  
  let p_low = 0.02425 in
  let p_high = 1.0 -. p_low in
  
  if Float.(p < 0.0 || p > 1.0) then
    invalid_arg "Probability must be between 0 and 1"
  else if Float.(p < p_low) then
    let q = Float.sqrt (-2.0 *. Float.log p) in
    (((((c1 *. q +. c2) *. q +. c3) *. q +. c4) *. q +. c5) *. q +. c6) /.
    ((((d1 *. q +. d2) *. q +. d3) *. q +. d4) *. q +. 1.0)
  else if Float.(p <= p_high) then
    let q = p -. 0.5 in
    let r = q *. q in
    (((((a1 *. r +. a2) *. r +. a3) *. r +. a4) *. r +. a5) *. r +. a6) *. q /.
    (((((b1 *. r +. b2) *. r +. b3) *. r +. b4) *. r +. b5) *. r +. 1.0)
  else
    let q = Float.sqrt (-2.0 *. Float.log (1.0 -. p)) in
    -.(((((c1 *. q +. c2) *. q +. c3) *. q +. c4) *. q +. c5) *. q +. c6) /.
    ((((d1 *. q +. d2) *. q +. d3) *. q +. d4) *. q +. 1.0)

let confidence_interval mean std_dev n alpha =
  let z = normal_quantile (1.0 -. alpha /. 2.0) in
  let margin = z *. std_dev /. Float.sqrt (Float.of_int n) in
  (mean -. margin, mean +. margin)

let t_test sample1 sample2 =
  let n1 = List.length sample1 in
  let n2 = List.length sample2 in
  let mean1 = List.fold sample1 ~init:0. ~f:(+.) /. Float.of_int n1 in
  let mean2 = List.fold sample2 ~init:0. ~f:(+.) /. Float.of_int n2 in
  let var1 = List.fold sample1 ~init:0. ~f:(fun acc x -> acc +. Float.((x - mean1) ** 2.)) /. Float.of_int (n1 - 1) in
  let var2 = List.fold sample2 ~init:0. ~f:(fun acc x -> acc +. Float.((x - mean2) ** 2.)) /. Float.of_int (n2 - 1) in
  let pooled_var = ((Float.of_int (n1 - 1) *. var1) +. (Float.of_int (n2 - 1) *. var2)) /. Float.of_int (n1 + n2 - 2) in
  let t_statistic = (mean1 -. mean2) /. Float.sqrt (pooled_var *. (1. /. Float.of_int n1 +. 1. /. Float.of_int n2)) in
  let df = n1 + n2 - 2 in
  (t_statistic, df)

let wilcoxon_test sample1 sample2 =
  let differences = List.map2_exn sample1 sample2 ~f:(fun x y -> x -. y) in
  let ranked_differences = 
    differences
    |> List.mapi ~f:(fun i d -> (i, Float.abs d))
    |> List.sort ~compare:(fun (_, d1) (_, d2) -> Float.compare d1 d2)
    |> List.mapi ~f:(fun rank (i, d) -> (i, d, Float.of_int (rank + 1)))
  in
  let w_plus = ranked_differences
    |> List.filter ~f:(fun (i, _, _) -> Float.(differences.(i) > 0.))
    |> List.fold ~init:0. ~f:(fun acc (_, _, rank) -> acc +. rank)
  in
  let n = List.length differences in
  let expected_w = Float.of_int (n * (n + 1)) /. 4. in
  let var_w = Float.of_int (n * (n + 1) * (2 * n + 1)) /. 24. in
  let z = (w_plus -. expected_w) /. Float.sqrt var_w in
  z

let friedman_test samples =
  let n = List.length (List.hd_exn samples) in
  let k = List.length samples in
  let ranks = List.map samples ~f:(fun sample ->
    sample
    |> List.mapi ~f:(fun i x -> (i, x))
    |> List.sort ~compare:(fun (_, x1) (_, x2) -> Float.compare x1 x2)
    |> List.mapi ~f:(fun rank (i, _) -> (i, Float.of_int (rank + 1)))
    |> List.sort ~compare:(fun (i1, _) (i2, _) -> Int.compare i1 i2)
    |> List.map ~f:snd
  ) in
  let rank_sums = List.map ranks ~f:(List.fold ~init:0. ~f:(+.)) in
  let chi_square = 12. *. Float.of_int n /. (Float.of_int k *. Float.of_int (k + 1)) *.
    (List.fold rank_sums ~init:0. ~f:(fun acc r -> acc +. r ** 2.)) -. 
    3. *. Float.of_int (n * (k + 1)) in
  (chi_square, k - 1)