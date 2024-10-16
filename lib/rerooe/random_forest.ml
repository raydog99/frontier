type tree = {
  feature: int;
  threshold: float;
  left: tree option;
  right: tree option;
  leaf_value: float option;
  quality: float;
}

type t = {
  trees: tree array;
  num_features: int;
}

let rec build_tree data features max_depth =
  if max_depth = 0 || List.length data <= 1 then
    { feature = -1; threshold = 0.; left = None; right = None; 
      leaf_value = Some (List.fold_left (fun acc (_, y) -> acc +. y) 0. data /. float_of_int (List.length data));
      quality = 0. }
  else
    match find_best_split data features with
    | None -> 
      { feature = -1; threshold = 0.; left = None; right = None; 
        leaf_value = Some (List.fold_left (fun acc (_, y) -> acc +. y) 0. data /. float_of_int (List.length data));
        quality = 0. }
    | Some { feature; threshold; quality } ->
      let left, right = List.partition (fun (x, _) -> x.(feature) <= threshold) data in
      { feature; threshold; 
        left = Some (build_tree left features (max_depth - 1));
        right = Some (build_tree right features (max_depth - 1));
        leaf_value = None;
        quality }

let find_best_split data features =
  let best_split = ref None in
  let best_quality = ref Float.neg_infinity in

  List.iter (fun feature ->
    let values = List.map (fun (x, _) -> x.(feature)) data in
    let sorted_values = List.sort_uniq compare values in
    List.iter (fun threshold ->
      let left, right = List.partition (fun (x, _) -> x.(feature) <= threshold) data in
      if List.length left > 0 && List.length right > 0 then
        let quality = calculate_split_quality left right in
        if quality > !best_quality then (
          best_quality := quality;
          best_split := Some { feature; threshold; quality }
        )
    ) sorted_values
  ) features;
  !best_split

let calculate_split_quality left right =
  let n_left = float_of_int (List.length left) in
  let n_right = float_of_int (List.length right) in
  let n_total = n_left +. n_right in

  let gini_impurity subset =
    let counts = Hashtbl.create 10 in
    List.iter (fun (_, y) ->
      Hashtbl.replace counts y (1 + (Hashtbl.find_opt counts y |> Option.value ~default:0))
    ) subset;
    let impurity = ref 1. in
    Hashtbl.iter (fun _ count ->
      let p = float_of_int count /. float_of_int (List.length subset) in
      impurity := !impurity -. (p *. p)
    ) counts;
    !impurity
  in

  let gini_left = gini_impurity left in
  let gini_right = gini_impurity right in
  let weighted_gini = (n_left /. n_total) *. gini_left +. (n_right /. n_total) *. gini_right in
  1. -. weighted_gini  (* Information gain *)

let create num_trees max_depth data =
  let num_features = Array.length (fst (List.hd data)) in
  let trees = Array.init num_trees (fun _ -> 
    build_tree data (List.init num_features (fun i -> i)) max_depth
  ) in
  { trees; num_features }

let rec predict_tree tree x =
  match tree.leaf_value with
  | Some v -> v
  | None ->
    if x.(tree.feature) <= tree.threshold then
      predict_tree (Option.get tree.left) x
    else
      predict_tree (Option.get tree.right) x

let predict rf x =
  let predictions = Array.map (fun tree -> predict_tree tree x) rf.trees in
  Array.fold_left (+.) 0. predictions /. float_of_int (Array.length rf.trees)

let rec update_feature_importance tree importances =
  match tree.leaf_value with
  | Some _ -> ()
  | None ->
      importances.(tree.feature) <- importances.(tree.feature) +. tree.quality;
      update_feature_importance (Option.get tree.left) importances;
      update_feature_importance (Option.get tree.right) importances

let feature_importance rf =
  let importances = Array.make rf.num_features 0. in
  Array.iter (fun tree -> update_feature_importance tree importances) rf.trees;
  let total_importance = Array.fold_left (+.) 0. importances in
  Array.map (fun imp -> imp /. total_importance) importances