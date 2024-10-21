type t = {
  weights: float array;
  n: int;
  volatility: float array array;
}

let create n initial_weights volatility =
  { weights = initial_weights; n; volatility }

let update market new_weights =
  { market with weights = new_weights }

let get_weight market i = market.weights.(i)
let get_weights market = market.weights
let size market = market.n
let get_volatility market = market.volatility

let normalize weights =
  let total = Array.fold_left (+.) 0.0 weights in
  Array.map (fun w -> w /. total) weights

let rank_based_diffusion initial_weights mu sigma gamma dt n_steps =
  let n = Array.length initial_weights in
  let generate_step weights =
    let ranked_weights = Array.mapi (fun i w -> (i, w)) weights
                         |> Array.to_list
                         |> List.sort (fun (_, w1) (_, w2) -> compare w2 w1) in
    let new_weights = Array.make n 0.0 in
    List.iteri (fun rank (i, w) ->
      let drift = mu *. (1.0 /. float_of_int (rank + 1) -. w) in
      let diffusion = sigma *. (w ** gamma) *. Random.gaussian() in
      new_weights.(i) <- w *. exp(drift *. dt +. diffusion *. sqrt dt)
    ) ranked_weights;
    normalize new_weights
  in
  let rec generate_sequence weights steps acc =
    if steps = 0 then List.rev acc
    else
      let new_weights = generate_step weights in
      let volatility = Array.make_matrix n n 0.0 in
      Array.iteri (fun i _ ->
        Array.iteri (fun j _ ->
          volatility.(i).(j) <- abs_float (new_weights.(i) -. weights.(i)) *. abs_float (new_weights.(j) -. weights.(j)) /. dt
        ) weights
      ) weights;
      let new_market = create n new_weights volatility in
      generate_sequence new_weights (steps - 1) (new_market :: acc)
  in
  generate_sequence initial_weights n_steps [create n initial_weights (Array.make_matrix n n 0.0)]

let diversity_coefficient market p =
  Array.fold_left (fun acc w -> acc +. (w ** p)) 0.0 market.weights
  |> fun x -> x ** (1.0 /. p)

let entropy market =
  Array.fold_left (fun acc w ->
    if w > 0.0 then acc -. w *. log w else acc
  ) 0.0 market.weights