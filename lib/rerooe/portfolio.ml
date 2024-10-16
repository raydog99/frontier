type asset = {
  model: Model.t;
  weight: float ref;
  sector: string;
}

type t = {
  assets: asset array;
  mutable total_value: float;
}

type performance_summary = {
  total_return: float;
  sharpe_ratio: float;
  max_drawdown: float;
}

let create models initial_weights =
  if Array.length models <> Array.length initial_weights then
    invalid_arg "Number of models and weights must match";
  let assets = Array.map2 (fun model weight -> { model; weight = ref weight; sector = Model.get_sector model }) models initial_weights in
  let total_value = Array.fold_left (fun acc asset -> acc +. !(asset.weight)) 0. in
  { assets; total_value }

let rebalance portfolio target_weights =
  if Array.length portfolio.assets <> Array.length target_weights then
    invalid_arg "Number of assets and target weights must match";
  Array.iteri (fun i asset ->
    asset.weight := target_weights.(i);
  ) portfolio.assets;
  portfolio.total_value <- Array.fold_left (fun acc weight -> acc +. weight) 0. target_weights

let update_asset_price portfolio symbol price volume =
  Array.iter (fun asset ->
    if Model.get_symbol asset.model = symbol then
      Model.update_price asset.model price volume
  ) portfolio.assets

let execute_order portfolio symbol quantity price =
  let asset = Array.find_opt (fun a -> Model.get_symbol a.model = symbol) portfolio.assets in
  match asset with
  | Some asset ->
      let current_position = Model.get_position asset.model in
      let new_position = current_position +. float_of_int quantity in
      Model.set_position asset.model new_position;
      let transaction_value = float_of_int quantity *. price in
      portfolio.total_value <- portfolio.total_value -. transaction_value;
      asset.weight := !asset.weight +. (transaction_value /. portfolio.total_value)
  | None -> failwith ("Asset not found: " ^ symbol)

let get_returns portfolio =
  Array.map (fun asset -> Model.get_return asset.model) portfolio.assets |> Array.to_list

let get_weights portfolio =
  Array.map (fun asset -> !(asset.weight)) portfolio.assets

let get_assets portfolio =
  portfolio.assets

let get_performance_summary portfolio =
  let returns = get_returns portfolio in
  let total_return = List.fold_left (+.) 0. returns in
  let avg_return = total_return /. float_of_int (List.length returns) in
  let variance = List.fold_left (fun acc r -> acc +. (r -. avg_return) ** 2.) 0. returns
                 /. float_of_int (List.length returns) in
  let std_dev = sqrt variance in
  let sharpe_ratio = if std_dev = 0. then 0. else avg_return /. std_dev in
  let max_drawdown = 
    let rec calc_max_drawdown peak drawdown = function
      | [] -> drawdown
      | hd :: tl ->
          if hd > peak then
            calc_max_drawdown hd drawdown tl
          else
            let current_drawdown = (peak -. hd) /. peak in
            calc_max_drawdown peak (max drawdown current_drawdown) tl
    in
    calc_max_drawdown 0. 0. returns
  in
  { total_return; sharpe_ratio; max_drawdown }

let get_total_value portfolio = portfolio.total_value

let get_asset_values portfolio =
  Array.map (fun asset -> !(asset.weight) *. Model.get_price asset.model) portfolio.assets