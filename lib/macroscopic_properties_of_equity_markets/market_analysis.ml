open Torch

let calculate_market_weights market_caps =
  let total_cap = Array.fold_left (+.) 0. market_caps in
  Array.map (fun cap -> cap /. total_cap) market_caps

let calculate_capital_distribution weights =
  let sorted_weights = Array.copy weights in
  Array.sort (fun a b -> compare b a) sorted_weights;
  sorted_weights

let calculate_shannon_entropy weights =
  Array.fold_left (fun acc w -> 
    if w > 0. then acc -. w *. log w /. log 2.
    else acc
  ) 0. weights

let calculate_excess_growth_rate weights_t0 weights_t1 =
  let n = Array.length weights_t0 in
  let sum_log_return = ref 0. in
  let sum_return = ref 0. in
  for i = 0 to n - 1 do
    let r = log (weights_t1.(i) /. weights_t0.(i)) in
    sum_log_return := !sum_log_return +. weights_t0.(i) *. r;
    sum_return := !sum_return +. weights_t0.(i) *. (exp r -. 1.)
  done;
  log (1. +. !sum_return) -. !sum_log_return

let calculate_rank_volatility market_caps =
  let n = Array.length market_caps in
  let m = Array.length market_caps.(0) in
  let rank_volatility = Array.make n 0. in
  for i = 0 to n - 1 do
    let sum_squared_diff = ref 0. in
    for j = 1 to m - 1 do
      let diff = log (market_caps.(i).(j) /. market_caps.(i).(j-1)) in
      sum_squared_diff := !sum_squared_diff +. diff *. diff
    done;
    rank_volatility.(i) <- sqrt (!sum_squared_diff /. float_of_int (m - 1))
  done;
  rank_volatility

let calculate_rank_transition_probabilities market_caps =
  let n = Array.length market_caps in
  let m = Array.length market_caps.(0) in
  let transition_matrix = Array.make_matrix n n 0. in
  for t = 1 to m - 1 do
    let ranks_t0 = Array.mapi (fun i cap -> (i, cap.(t-1))) market_caps
                   |> Array.to_list
                   |> List.sort (fun (_, a) (_, b) -> compare b a)
                   |> List.mapi (fun rank (i, _) -> (i, rank)) in
    let ranks_t1 = Array.mapi (fun i cap -> (i, cap.(t))) market_caps
                   |> Array.to_list
                   |> List.sort (fun (_, a) (_, b) -> compare b a)
                   |> List.mapi (fun rank (i, _) -> (i, rank)) in
    List.iter2 (fun (i, rank_t0) (j, rank_t1) ->
      transition_matrix.(rank_t0).(rank_t1) <- transition_matrix.(rank_t0).(rank_t1) +. 1.
    ) ranks_t0 ranks_t1
  done;
  Array.map (fun row ->
    let sum = Array.fold_left (+.) 0. row in
    Array.map (fun x -> x /. sum) row
  ) transition_matrix

let calculate_rank_switching_intensity market_caps =
  let n = Array.length market_caps in
  let m = Array.length market_caps.(0) in
  let intensity = Array.make n 0. in
  for t = 1 to m - 1 do
    let ranks_t0 = Array.mapi (fun i cap -> (i, cap.(t-1))) market_caps
                   |> Array.to_list
                   |> List.sort (fun (_, a) (_, b) -> compare b a)
                   |> List.mapi (fun rank (i, _) -> (i, rank)) in
    let ranks_t1 = Array.mapi (fun i cap -> (i, cap.(t))) market_caps
                   |> Array.to_list
                   |> List.sort (fun (_, a) (_, b) -> compare b a)
                   |> List.mapi (fun rank (i, _) -> (i, rank)) in
    List.iter2 (fun (i, rank_t0) (j, rank_t1) ->
      let diff = abs (rank_t1 - rank_t0) in
      intensity.(rank_t0) <- intensity.(rank_t0) +. float_of_int diff
    ) ranks_t0 ranks_t1
  done;
  Array.map (fun x -> x /. float_of_int (m - 1)) intensity

let calculate_market_weights_tensor market_caps =
  let total_cap = Tensor.sum market_caps ~dim:[1] ~keepdim:true in
  Tensor.div market_caps total_cap

let calculate_capital_distribution_tensor weights =
  Tensor.sort weights ~descending:true ~dim:1 |> fst

let calculate_rank_volatility_tensor market_caps =
  let log_returns = Tensor.log (Tensor.div market_caps (Tensor.roll market_caps 1 ~dims:[0])) in
  Tensor.std log_returns ~dim:[0]

let calculate_rank_transition_probabilities_tensor market_caps =
  let n, m = Tensor.shape2_exn market_caps in
  let ranks = Tensor.argsort market_caps ~dim:1 ~descending:true in
  let transitions = Tensor.zeros [n - 1; m; m] in
  for t = 0 to n - 2 do
    let ranks_t = Tensor.select ranks ~dim:0 ~index:t in
    let ranks_t1 = Tensor.select ranks ~dim:0 ~index:(t + 1) in
    Tensor.index_put_ transitions [t; ranks_t; ranks_t1] (Tensor.ones [m])
  done;
  Tensor.div transitions (Tensor.sum transitions ~dim:[0] ~keepdim:true)

let get_rank market_caps index =
  let sorted_indices = 
    Array.init (Array.length market_caps) (fun i -> i)
    |> Array.sort (fun i j -> compare market_caps.(j) market_caps.(i))
  in
  Array.index_of ((=) index) sorted_indices

let get_top_k_stocks market_caps k =
  let sorted_indices = 
    Array.init (Array.length market_caps) (fun i -> i)
    |> Array.sort (fun i j -> compare market_caps.(j) market_caps.(i))
  in
  Array.to_list (Array.sub sorted_indices 0 k)