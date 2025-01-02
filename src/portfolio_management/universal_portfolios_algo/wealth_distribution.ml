type t = {
  distribution: (Portfolio.t * float) list;
}

let create portfolios =
  let n = List.length portfolios in
  let initial_weight = 1.0 /. float_of_int n in
  let create_portfolio p =
    let initial_market = Market.create n (Array.make n (1.0 /. float_of_int n)) (Array.make_matrix n n 0.0) in
    (Portfolio.create n (FunctionallyGeneratedPortfolio.generate_weights p initial_market), initial_weight)
  in
  { distribution = List.map create_portfolio portfolios }

let update distribution market_seq t =
  let total_wealth = List.fold_left (fun acc (p, _) -> 
    acc +. Portfolio.relative_value p market_seq t
  ) 0.0 distribution.distribution in
  let new_distribution = List.map (fun (p, _) ->
    let rv = Portfolio.relative_value p market_seq t in
    (p, rv /. total_wealth)
  ) distribution.distribution in
  { distribution = new_distribution }

let get_distribution t = t.distribution

let covers_universal_portfolio distribution market_seq t =
  let weights = List.map (fun (p, w) ->
    Array.map (fun pw -> pw *. w) (Portfolio.get_weights p)
  ) distribution.distribution in
  let combined_weights = List.fold_left (Array.map2 (+.)) 
    (Array.make (Market.size (List.hd market_seq)) 0.0) weights in
  Portfolio.create (Array.length combined_weights) combined_weights
  |> Portfolio.normalize

let wealth_concentration distribution =
  let sorted_weights = List.sort (fun (_, w1) (_, w2) -> compare w2 w1) distribution.distribution in
  let total_wealth = List.fold_left (fun acc (_, w) -> acc +. w) 0.0 sorted_weights in
  let rec calculate_concentration acc wealth_sum = function
    | [] -> acc
    | (_, w) :: rest ->
        let new_wealth_sum = wealth_sum +. w in
        if new_wealth_sum > 0.5 *. total_wealth then
          acc + 1
        else
          calculate_concentration (acc + 1) new_wealth_sum rest
  in
  calculate_concentration 0 0.0 sorted_weights

let wealth_entropy distribution =
  List.fold_left (fun acc (_, w) ->
    if w > 0.0 then acc -. w *. log w else acc
  ) 0.0 distribution.distribution