type t = {
    weights: float array;
    n: int;
  }

let create n initial_weights =
    { weights = initial_weights; n }

let update portfolio new_weights =
    { portfolio with weights = new_weights }

let get_weight portfolio i = portfolio.weights.(i)
let get_weights portfolio = portfolio.weights
let size portfolio = portfolio.n

let relative_value portfolio market_seq t =
    let rec calculate_value acc t =
      if t = 0 then acc
      else
        let market = List.nth market_seq t in
        let prev_market = List.nth market_seq (t - 1) in
        let ratio = Array.map2 (fun m pm -> m /. pm) (Market.get_weights market) (Market.get_weights prev_market) in
        let new_acc = acc *. Array.fold_left2 (fun acc p r -> acc +. p *. r) 0.0 portfolio.weights ratio in
        calculate_value new_acc (t - 1)
    in
    calculate_value 1.0 t

let log_relative_value portfolio market_seq t =
    log (relative_value portfolio market_seq t)

let normalize portfolio =
    let total = Array.fold_left (+.) 0.0 portfolio.weights in
    { portfolio with weights = Array.map (fun w -> w /. total) portfolio.weights }