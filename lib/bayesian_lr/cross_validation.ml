open Torch
open Type

let create_folds x y k =
  let n = size x 0 in
  let fold_size = n / k in
  let indices = randperm n in
  
  Array.init k (fun i ->
    let start_idx = i * fold_size in
    let end_idx = if i = k - 1 then n else (i + 1) * fold_size in
    
    (* Test indices for this fold *)
    let test_idx = narrow indices 0 start_idx (end_idx - start_idx) in
    
    (* Train indices (everything else) *)
    let train_idx = cat [
      narrow indices 0 0 start_idx;
      narrow indices 0 end_idx (n - end_idx)
    ] 0 in
    
    (* Create fold data *)
    let test_x = index_select x 0 test_idx in
    let test_y = index_select y 0 test_idx in
    let train_x = index_select x 0 train_idx in
    let train_y = index_select y 0 train_idx in
    
    {train_x; train_y; test_x; test_y}
  )

let grid_search x y config k lambda_grid =
  let folds = create_folds x y k in
  
  (* Test each lambda value *)
  let results = List.map (fun lambda ->
    let fold_errors = Array.map (fun fold -