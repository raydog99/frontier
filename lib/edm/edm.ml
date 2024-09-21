open Torch

type stock = {
  id: string;
  returns: Tensor.t;
  sector: string;
}

type network = {
  stocks: stock list;
  edges: (string * string) list;
}

type risk_measure = VaR | ES

type community = stock list

let tensor_norm tensor =
  Tensor.norm tensor ~p:Scalar.Two ~dim:[0] ~keepdim:true

let calculate_edm stock1 stock2 =
  let r1 = stock1.returns in
  let r2 = stock2.returns in
  let n = Tensor.shape r1 |> List.hd |> float_of_int in
  
  let norm1 = tensor_norm r1 in
  let norm2 = tensor_norm r2 in
  
  let normalized1 = Tensor.div r1 norm1 in
  let normalized2 = Tensor.div r2 norm2 in
  
  let product = Tensor.mul normalized1 normalized2 in
  let sum = Tensor.sum product ~dim:[0] ~keepdim:true in
  
  Tensor.div sum (Tensor.float_vec [n])

let construct_network stocks threshold =
  let edges = 
    List.concat_map (fun s1 ->
      List.filter_map (fun s2 ->
        if s1.id < s2.id then
          let edm = calculate_edm s1 s2 in
          if Tensor.get_float0_exn edm >= threshold then
            Some (s1.id, s2.id)
          else
            None
        else
          None
      ) stocks
    ) stocks
  in
  { stocks = stocks; edges = edges }

let is_connected_to_set stock set edges =
  List.exists (fun s ->
    List.mem (stock.id, s.id) edges || List.mem (s.id, stock.id) edges
  ) set

let find_max_independent_set network =
  let rec build_set remaining_stocks set =
    match remaining_stocks with
    | [] -> set
    | stock :: rest ->
      if is_connected_to_set stock set network.edges then
        build_set rest set
      else
        build_set rest (stock :: set)
  in
  build_set network.stocks []

let calculate_var returns confidence_level =
  let sorted_returns = Tensor.sort returns ~descending:false ~stable:true in
  let index = int_of_float (float (Tensor.shape returns |> List.hd) *. (1.0 -. confidence_level)) in
  Tensor.get sorted_returns [|index|]

let calculate_es returns confidence_level =
  let sorted_returns = Tensor.sort returns ~descending:false ~stable:true in
  let index = int_of_float (float (Tensor.shape returns |> List.hd) *. (1.0 -. confidence_level)) in
  let tail = Tensor.narrow sorted_returns ~dim:0 ~start:0 ~length:index in
  Tensor.mean tail

let calculate_portfolio_return weights returns =
  Tensor.sum (Tensor.mul weights returns)

let calculate_portfolio_risk weights returns risk_measure confidence_level =
  let portfolio_returns = calculate_portfolio_return weights returns in
  match risk_measure with
  | VaR -> calculate_var portfolio_returns confidence_level
  | ES -> calculate_es portfolio_returns confidence_level

let optimize_portfolio stocks risk_measure confidence_level target_return =
  let n = List.length stocks in
  let returns = Tensor.stack (List.map (fun s -> s.returns) stocks) ~dim:0 in
  
  let optimize weights =
    let risk = calculate_portfolio_risk weights returns risk_measure confidence_level in
    let return = calculate_portfolio_return weights returns in
    let constraint_penalty = 
      Tensor.relu (Tensor.sub (Tensor.float_vec [target_return]) return) |>
      Tensor.mul (Tensor.float_vec [1000.0])
    in
    Tensor.add risk constraint_penalty
  in
  
  let weights = Tensor.ones [n] |> Tensor.div_scalar (Scalar.float (float_of_int n)) in
  let weights = Tensor.set_requires_grad weights true in
  
  let optimizer = Optimizer.adam [weights] ~lr:0.01 in
  
  for _ = 1 to 1000 do
    Optimizer.zero_grad optimizer;
    let loss = optimize weights in
    Tensor.backward loss;
    Optimizer.step optimizer;
  done;
  
  Tensor.to_float1 weights |> Array.to_list

let load_stock_data filename =
  let ic = open_in filename in
  let rec read_lines acc =
    try
      let line = input_line ic in
      let parts = String.split_on_char ',' line in
      match parts with
      | id :: sector :: returns ->
        let returns_tensor = 
          returns 
          |> List.map float_of_string 
          |> Tensor.of_float1
        in
        let stock = { id; returns = returns_tensor; sector } in
        read_lines (stock :: acc)
    with End_of_file ->
      close_in ic;
      List.rev acc
  in
  read_lines []
end

module Analysis = struct
open Edm

let betweenness_centrality network =
  let n = List.length network.stocks in
  let centrality = Hashtbl.create n in
  List.iter (fun stock -> Hashtbl.add centrality stock.id 0.0) network.stocks;

  let bfs source =
    let queue = Queue.create () in
    let distances = Hashtbl.create n in
    let paths = Hashtbl.create n in
    Queue.push source queue;
    Hashtbl.add distances source 0;
    Hashtbl.add paths source 1;

    while not (Queue.is_empty queue) do
      let v = Queue.pop queue in
      let neighbors = List.filter (fun (s1, s2) -> s1 = v || s2 = v) network.edges in
      List.iter (fun (s1, s2) ->
        let w = if s1 = v then s2 else s1 in
        if not (Hashtbl.mem distances w) then begin
          Queue.push w queue;
          Hashtbl.add distances w ((Hashtbl.find distances v) + 1);
          Hashtbl.add paths w (Hashtbl.find paths v)
        end else if (Hashtbl.find distances w) = ((Hashtbl.find distances v) + 1) then
          Hashtbl.replace paths w ((Hashtbl.find paths w) + (Hashtbl.find paths v))
      ) neighbors
    done;
    (distances, paths)
  in

  List.iter (fun source ->
    let (distances, paths) = bfs source.id in
    let credits = Hashtbl.create n in
    List.iter (fun stock -> Hashtbl.add credits stock.id 1.0) network.stocks;
    List.sort (fun s1 s2 -> compare (Hashtbl.find distances s2.id) (Hashtbl.find distances s1.id)) network.stocks
    |> List.iter (fun v ->
      let neighbors = List.filter (fun (s1, s2) -> (s1 = v.id && s2 <> source.id) || (s2 = v.id && s1 <> source.id)) network.edges in
      List.iter (fun (s1, s2) ->
        let w = if s1 = v.id then s2 else s1 in
        if (Hashtbl.find distances w) = ((Hashtbl.find distances v.id) - 1) then begin
          let credit = (Hashtbl.find credits v.id) *. (float_of_int (Hashtbl.find paths w)) /. (float_of_int (Hashtbl.find paths v.id)) in
          Hashtbl.replace credits w ((Hashtbl.find credits w) +. credit);
          if v.id <> source.id then
            Hashtbl.replace centrality v.id ((Hashtbl.find centrality v.id) +. credit)
        end
      ) neighbors
    )
  ) network.stocks;
  centrality

let detect_communities network =
  let rec remove_edge network =
    let centrality = betweenness_centrality network in
    let max_edge = List.fold_left (fun acc (s1, s2) ->
      let edge_centrality = (Hashtbl.find centrality s1) +. (Hashtbl.find centrality s2) in
      if edge_centrality > (snd acc) then ((s1, s2), edge_centrality) else acc
    ) (("", ""), 0.0) network.edges in
    let updated_edges = List.filter (fun e -> e <> (fst max_edge)) network.edges in
    let updated_network = { network with edges = updated_edges } in
    
    let rec dfs visited node =
      let neighbors = List.filter (fun (s1, s2) -> s1 = node || s2 = node) updated_network.edges in
      List.iter (fun (s1, s2) ->
        let next = if s1 = node then s2 else s1 in
        if not (Hashtbl.mem visited next) then begin
          Hashtbl.add visited next ();
          dfs visited next
        end
      ) neighbors
    in
    
    let communities = ref [] in
    let all_visited = Hashtbl.create (List.length network.stocks) in
    List.iter (fun stock ->
      if not (Hashtbl.mem all_visited stock.id) then begin
        let community_visited = Hashtbl.create (List.length network.stocks) in
        Hashtbl.add community_visited stock.id ();
        dfs community_visited stock.id;
        let community = Hashtbl.fold (fun id _ acc -> 
          List.find (fun s -> s.id = id) network.stocks :: acc
        ) community_visited [] in
        communities := community :: !communities;
        Hashtbl.iter (fun id _ -> Hashtbl.add all_visited id ()) community_visited
      end
    ) network.stocks;
    
    if List.length !communities > 1 then !communities
    else remove_edge updated_network
  in
  remove_edge network

let calculate_modularity network communities =
  let m = float_of_int (List.length network.edges) in
  let modularity = ref 0.0 in
  List.iter (fun community ->
    List.iter (fun v ->
      List.iter (fun w ->
        let a_vw = if List.mem (v.id, w.id) network.edges || List.mem (w.id, v.id) network.edges then 1.0 else 0.0 in
        let k_v = float_of_int (List.length (List.filter (fun (s1, s2) -> s1 = v.id || s2 = v.id) network.edges)) in
        let k_w = float_of_int (List.length (List.filter (fun (s1, s2) -> s1 = w.id || s2 = w.id) network.edges)) in
        modularity := !modularity +. (a_vw -. (k_v *. k_w) /. (2.0 *. m))
      ) community
    ) community
  ) communities;
  !modularity /. (2.0 *. m)

let network_statistics network =
  let n = List.length network.stocks in
  let m = List.length network.edges in
  let avg_degree = (2.0 *. float_of_int m) /. float_of_int n in
  let degrees = List.map (fun stock ->
    List.length (List.filter (fun (s1, s2) -> s1 = stock.id || s2 = stock.id) network.edges)
  ) network.stocks in
  let max_degree = List.fold_left max 0 degrees in
  let min_degree = List.fold_left min max_int degrees in
  {
    num_nodes = n;
    num_edges = m;
    avg_degree = avg_degree;
    max_degree = max_degree;
    min_degree = min_degree;
  }

let descriptive_statistics returns =
  let n = Tensor.shape returns |> List.hd |> float_of_int in
  let mean = Tensor.mean returns |> Tensor.get_float0_exn in
  let std = Tensor.std returns ~unbiased:true |> Tensor.get_float0_exn in
  let sorted = Tensor.sort returns ~descending:false ~stable:true in
  let median = 
    if int_of_float n mod 2 = 0 then
      (Tensor.get_float1 sorted (int_of_float n / 2 - 1) +. Tensor.get_float1 sorted (int_of_float n / 2)) /. 2.0
    else
      Tensor.get_float1 sorted (int_of_float n / 2)
  in
  let min = Tensor.get_float1 sorted 0 in
  let max = Tensor.get_float1 sorted (int_of_float n - 1) in
  {
    mean = mean;
    median = median;
    std = std;
    min = min;
    max = max;
    skewness = Tensor.mean (Tensor.pow_tensor_scalar (Tensor.sub returns (Tensor.float_vec [mean])) (Tensor.float_vec [3.0])) 
              |> Tensor.div_scalar (Scalar.float (std ** 3.0))
              |> Tensor.get_float0_exn;
    kurtosis = Tensor.mean (Tensor.pow_tensor_scalar (Tensor.sub returns (Tensor.float_vec [mean])) (Tensor.float_vec [4.0]))
              |> Tensor.div_scalar (Scalar.float (std ** 4.0))
              |> Tensor.sub_scalar (Scalar.float 3.0)
              |> Tensor.get_float0_exn;
  }

let classify_by_sector stocks =
  List.fold_left (fun acc stock ->
    let sector_stocks = 
      try List.assoc stock.sector acc 
      with Not_found -> []
    in
    (stock.sector, stock :: sector_stocks) :: 
    List.remove_assoc stock.sector acc
  ) [] stocks

let calculate_portfolio_performance weights stocks start_date end_date =
  let returns = Tensor.stack (List.map (fun s -> s.returns) stocks) ~dim:0 in
  let weights_tensor = Tensor.of_float1 (Array.of_list weights) in
  let portfolio_returns = Tensor.matmul weights_tensor returns in
  let cumulative_returns = Tensor.cumprod portfolio_returns ~dim:0 ~exclusive:false in
  Tensor.to_float1 cumulative_returns |> Array.to_list

let backtest_portfolio strategy stocks risk_measure confidence_level target_return start_date end_date =
  let weights = strategy stocks risk_measure confidence_level target_return in
  calculate_portfolio_performance weights stocks start_date end_date

let compare_strategies strategies stocks risk_measure confidence_level target_return start_date end_date =
  List.map (fun strategy ->
    let performance = backtest_portfolio strategy stocks risk_measure confidence_level target_return start_date end_date in
    (strategy, performance)
  ) strategies

let optimize_portfolio_subset stocks subset risk_measure confidence_level target_return =
  let subset_stocks = List.filter (fun s -> List.mem s subset) stocks in
  Edm.optimize_portfolio subset_stocks risk_measure confidence_level target_return

let create_combined_strategy community_strategy sector_strategy stocks risk_measure confidence_level target_return =
  let communities = detect_communities { stocks = stocks; edges = [] } in
  let sectors = classify_by_sector stocks in
  
  let community_weights = List.map (fun community ->
    community_strategy community risk_measure confidence_level target_return
  ) communities in
  
  let sector_weights = List.map (fun (_, sector_stocks) ->
    sector_strategy sector_stocks risk_measure confidence_level target_return
  ) sectors in
  
  let combined_weights = List.map2 (fun cw sw -> 
    List.map2 (fun a b -> (a +. b) /. 2.0) cw sw
  ) community_weights sector_weights in
  
  List.flatten combined_weights