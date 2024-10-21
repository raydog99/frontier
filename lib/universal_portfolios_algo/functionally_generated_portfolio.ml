type t = {
  generate: Market.t -> float array;
  name: string;
  generating_function: Market.t -> float;
}

let create name generate_fn generating_function =
  { generate = generate_fn; name; generating_function }

let generate_weights portfolio market = portfolio.generate market
let get_name portfolio = portfolio.name
let get_generating_function portfolio = portfolio.generating_function

let constant_weighted weights =
  let generating_function market =
    Array.fold_left2 (fun acc w m -> acc +. w *. log m) 0.0 weights (Market.get_weights market)
  in
  create "Constant-weighted" (fun _ -> weights) generating_function

let diversity_weighted p =
  let generating_function market =
    let weights = Market.get_weights market in
    (Array.fold_left (fun acc w -> acc +. w ** p) 0.0 weights) ** (1.0 /. p)
  in
  create (Printf.sprintf "Diversity-weighted (p=%.2f)" p)
    (fun market ->
      let weights = Market.get_weights market in
      Array.map (fun w -> w ** p) weights)
    generating_function

let entropy_weighted =
  let generating_function = Market.entropy in
  create "Entropy-weighted"
    (fun market ->
      let weights = Market.get_weights market in
      Array.map (fun w -> -. w *. log w) weights)
    generating_function

let equal_weighted n =
  let generating_function market =
    let weights = Market.get_weights market in
    Array.fold_left (fun acc w -> acc -. (1.0 /. float_of_int n) *. log w) 0.0 weights
  in
  create "Equal-weighted"
    (fun _ -> Array.make n (1.0 /. float_of_int n))
    generating_function

let market_weighted =
  let generating_function market =
    let weights = Market.get_weights market in
    Array.fold_left (fun acc w -> acc +. w *. log w) 0.0 weights
  in
  create "Market-weighted" Market.get_weights generating_function

let diversity_weighted_generalized s p =
  let generating_function market =
    let weights = Market.get_weights market in
    let n = Array.length weights in
    let ranked_weights = Array.mapi (fun i w -> (i, w)) weights
                         |> Array.to_list
                         |> List.sort (fun (_, w1) (_, w2) -> compare w2 w1) in
    List.fold_left (fun acc (rank, w) ->
      acc +. (w ** p) *. (float_of_int (rank + 1) ** (s *. (1.0 -. p)))
    ) 0.0 ranked_weights
  in
  create (Printf.sprintf "Generalized diversity-weighted (s=%.2f, p=%.2f)" s p)
    (fun market ->
      let weights = Market.get_weights market in
      let n = Array.length weights in
      let ranked_weights = Array.mapi (fun i w -> (i, w)) weights
                           |> Array.to_list
                           |> List.sort (fun (_, w1) (_, w2) -> compare w2 w1) in
      List.mapi (fun rank (i, w) ->
        w ** p *. (float_of_int (rank + 1) ** (s *. (1.0 -. p)))
      ) ranked_weights
      |> Array.of_list)
    generating_function

let volatility_weighted alpha =
  let generating_function market =
    let weights = Market.get_weights market in
    let volatility = Market.get_volatility market in
    Array.fold_left2 (fun acc w v ->
      acc +. (w ** (1.0 -. alpha)) *. (sqrt v ** (-. alpha))
    ) 0.0 weights (Array.map (fun row -> row.(0)) volatility)
  in
  create (Printf.sprintf "Volatility-weighted (alpha=%.2f)" alpha)
    (fun market ->
      let weights = Market.get_weights market in
      let volatility = Market.get_volatility market in
      Array.mapi (fun i w ->
        (w ** (1.0 -. alpha)) *. (sqrt volatility.(i).(i) ** (-. alpha))
      ) weights
    )
    generating_function

let adaptive_boltzmann T eta =
  let generating_function market =
    let weights = Market.get_weights market in
    Array.fold_left (fun acc w ->
      acc +. exp (w /. T)
    ) 0.0 weights
  in
  create (Printf.sprintf "Adaptive Boltzmann (T=%.2f, eta=%.2f)" T eta)
    (fun market ->
      let weights = Market.get_weights market in
      let boltzmann_weights = Array.map (fun w -> exp (w /. T)) weights in
      Array.map2 (fun bw w -> (bw ** (1.0 -. eta)) *. (w ** eta)) boltzmann_weights weights
    )
    generating_function

let rank_dependent alpha beta =
  let generating_function market =
    let weights = Market.get_weights market in
    let n = Array.length weights in
    let ranked_weights = Array.mapi (fun i w -> (i, w)) weights
                         |> Array.to_list
                         |> List.sort (fun (_, w1) (_, w2) -> compare w2 w1) in
    List.fold_left (fun acc (rank, w) ->
      acc +. w ** alpha *. (float_of_int (rank + 1) ** (-. beta))
    ) 0.0 ranked_weights
  in
  create (Printf.sprintf "Rank-dependent (alpha=%.2f, beta=%.2f)" alpha beta)
    (fun market ->
      let weights = Market.get_weights market in
      let n = Array.length weights in
      let ranked_weights = Array.mapi (fun i w -> (i, w)) weights
                           |> Array.to_list
                           |> List.sort (fun (_, w1) (_, w2) -> compare w2 w1) in
      List.mapi (fun rank (i, w) ->
        w ** alpha *. (float_of_int (rank + 1) ** (-. beta))
      ) ranked_weights
      |> Array.of_list
    )
    generating_function

let cross_entropy alpha =
  let generating_function market =
    let weights = Market.get_weights market in
    Array.fold_left (fun acc w ->
      acc -. w *. log (w ** alpha)
    ) 0.0 weights
  in
  create (Printf.sprintf "Cross-entropy (alpha=%.2f)" alpha)
    (fun market ->
      let weights = Market.get_weights market in
      Array.map (fun w -> w ** (1.0 -. alpha)) weights
    )
    generating_function

let log_barrier c =
  let generating_function market =
    let weights = Market.get_weights market in
    Array.fold_left (fun acc w ->
      acc -. c *. log w
    ) 0.0 weights
  in
  create (Printf.sprintf "Log-barrier (c=%.2f)" c)
    (fun market ->
      let weights = Market.get_weights market in
      Array.map (fun w -> c /. w) weights
    )
    generating_function

let relative_entropy p q =
  Array.fold_left2 (fun acc pi qi ->
    if pi > 0.0 then acc +. pi *. log (pi /. qi) else acc
  ) 0.0 p q

let supergradient_vector portfolio market =
  let weights = Market.get_weights market in
  let n = Array.length weights in
  let epsilon = 1e-6 in
  Array.init n (fun i ->
    let perturbed_weights = Array.copy weights in
    perturbed_weights.(i) <- perturbed_weights.(i) +. epsilon;
    let perturbed_market = Market.create n perturbed_weights (Market.get_volatility market) in
    (portfolio.generating_function perturbed_market -. portfolio.generating_function market) /. epsilon
  )