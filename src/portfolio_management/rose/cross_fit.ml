let create_folds n k =
  let fold_size = n / k in
  let indices = Array.init n (fun i -> i) in
  Array.shuffle indices;
  Array.init k (fun i ->
    let test_start = i * fold_size in
    let test_indices = Array.sub indices test_start fold_size in
    let train_indices = Array.concat [
      Array.sub indices 0 test_start;
      Array.sub indices (test_start + fold_size) (n - test_start - fold_size)
    ] in
    { train_indices; test_indices }
  )

let create_stratified_folds data n_folds =
  let n = Array.length data in
  let sorted_indices = Array.init n (fun i -> i) in
  Array.sort (fun i j -> compare data.(i).y data.(j).y) sorted_indices;
  
  let fold_size = n / n_folds in
  let folds = Array.init n_folds (fun _ -> [||]) in
  
  Array.iteri (fun i idx ->
    let fold = i mod n_folds in
    folds.(fold) <- Array.append folds.(fold) [|idx|]
  ) sorted_indices;
  
  Array.map (fun test_indices ->
    let train_indices = Array.concat (
      Array.to_list (Array.filteri (fun i f -> i <> Array.length folds) folds)
    ) in
    { train_indices; test_indices }
  ) folds

let cross_fit data n_folds model_type =
  let folds = create_stratified_folds data n_folds in
  let n = Array.length data in
  let nuisance_estimates = Array.make n {f_hat = 0.0; m_hat = 0.0} in
  
  Array.iter (fun fold ->
    let train_data = Array.map (fun i -> data.(i)) fold.train_indices in
    let test_data = Array.map (fun i -> data.(i)) fold.test_indices in
    let train_nuisance = NuisanceEstimator.estimate_all train_data model_type in
    Array.iteri (fun i idx ->
      nuisance_estimates.(idx) <- NuisanceEstimator.predict test_data.(i) train_nuisance
    ) fold.test_indices
  ) folds;
  
  (folds, nuisance_estimates)