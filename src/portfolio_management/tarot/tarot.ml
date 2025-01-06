open Torch

type dataset = {
  features: Tensor.t;
  labels: Tensor.t;
}

type gradient_features = {
  features: Tensor.t;
  checkpoints: int;
}

type transport_plan = {
  coupling: Tensor.t;
  cost: float;
}

module Checkpoint = struct
  type t = {
    model_state: (string * Tensor.t) list;
    optimizer_state: (string * Tensor.t) list;
    iteration: int;
    learning_rate: float;
  }

  let save checkpoint path =
    let state = Tensor.Dict.create () in
    List.iter (fun (name, tensor) ->
      Tensor.Dict.set state ~key:("model." ^ name) tensor
    ) checkpoint.model_state;
    List.iter (fun (name, tensor) ->
      Tensor.Dict.set state ~key:("optim." ^ name) tensor
    ) checkpoint.optimizer_state;
    Tensor.Dict.set state ~key:"iteration" (Tensor.of_int0 checkpoint.iteration);
    Tensor.Dict.set state ~key:"learning_rate" (Tensor.of_float0 checkpoint.learning_rate);
    Tensor.save state path

  let load path =
    let state = Tensor.load path in
    let model_state = ref [] in
    let optimizer_state = ref [] in
    let iteration = ref 0 in
    let learning_rate = ref 0.0 in
    
    Tensor.Dict.iter state ~f:(fun ~key ~data ->
      match key with
      | key when String.starts_with ~prefix:"model." key ->
        let name = String.sub key 6 (String.length key - 6) in
        model_state := (name, data) :: !model_state
      | key when String.starts_with ~prefix:"optim." key ->
        let name = String.sub key 6 (String.length key - 6) in
        optimizer_state := (name, data) :: !optimizer_state
      | "iteration" -> iteration := Tensor.to_int0_exn data
      | "learning_rate" -> learning_rate := Tensor.to_float0_exn data
      | _ -> ()
    );
    
    {
      model_state = List.rev !model_state;
      optimizer_state = List.rev !optimizer_state;
      iteration = !iteration;
      learning_rate = !learning_rate;
    }

  let load_model model checkpoint =
    List.iter (fun (name, tensor) ->
      Model.load_state_dict model [name, tensor]
    ) checkpoint.model_state

  let create_ensemble ?(n_models=3) model =
    List.init n_models (fun _ ->
      Model.copy model
    )
end

module DataInfluence = struct
  type influence_config = {
    checkpoint_interval: int;
    max_checkpoints: int;
    lr_schedule: (int * float) list;
    gradient_norm_clip: float option;
    device: Device.t;
  }

  let default_config = {
    checkpoint_interval = 100;
    max_checkpoints = 10;
    lr_schedule = [(0, 0.1); (1000, 0.01); (2000, 0.001)];
    gradient_norm_clip = Some 1.0;
    device = Device.Cpu;
  }

  let compute_batch_gradients model batch checkpoint =
    let predicted = Model.forward model batch.features in
    let divergence = Divergence.cross_entropy predicted batch.labels in
    Tensor.backward divergence;
    Tensor.grad batch.features

  let compute_gradient_checkpoints ?(config=default_config) model dataset =
    let checkpoints = ref [] in
    let features_acc = ref None in
    
    let get_lr iteration =
      let rec find_lr = function
        | [] -> 0.001
        | [(step, lr)] -> lr
        | (step1, lr1)::(step2, lr2)::_ when iteration < step2 -> lr1
        | _::rest -> find_lr rest
      in
      find_lr config.lr_schedule
    in
    
    for i = 0 to config.max_checkpoints - 1 do
      let lr = get_lr (i * config.checkpoint_interval) in
      let checkpoint = {
        Checkpoint.model_state = Model.state_dict model;
        optimizer_state = [];
        iteration = i * config.checkpoint_interval;
        learning_rate = lr;
      } in
      checkpoints := checkpoint :: !checkpoints
    done;
    
    List.rev !checkpoints

  let compute_influence xi x checkpoints =
    List.fold_left (fun acc checkpoint ->
      let xi_grad = compute_batch_gradients checkpoint.model_state xi checkpoint in
      let x_grad = compute_batch_gradients checkpoint.model_state x checkpoint in
      let influence = Tensor.dot 
        (Tensor.reshape xi_grad [-1]) 
        (Tensor.reshape x_grad [-1]) in
      Tensor.add acc (Tensor.mul_scalar influence (-1.0 *. checkpoint.learning_rate))
    ) (Tensor.zeros []) checkpoints
end

module GradientFeatures = struct
  type embedding_config = {
    feature_dim: int;
    compression_ratio: float;
    batch_size: int;
    n_workers: int;
    device: Device.t;
    use_mixed_precision: bool;
    cache_dir: string option;
  }

  let default_config = {
    feature_dim = 512;
    compression_ratio = 0.1;
    batch_size = 32;
    n_workers = 4;
    device = Device.Cpu;
    use_mixed_precision = false;
    cache_dir = None;
  }

  let process_batch model batch checkpoint use_mixed_precision =
    if use_mixed_precision then
      Torch_core.with_autocast (fun () ->
        DataInfluence.compute_batch_gradients model batch checkpoint
      )
    else
      DataInfluence.compute_batch_gradients model batch checkpoint

  let embed_gradients ?(config=default_config) model checkpoints dataset =
    
    (* Check cache first *)
    match config.cache_dir with
    | Some dir ->
        let cache_path = Filename.concat dir "gradient_features.pt" in
        if Sys.file_exists cache_path then
          load cache_path
        else begin
          (* Process and cache *)
          let n_samples = shape2_exn dataset.features |> fst in
          let target_dim = int_of_float (float_of_int config.feature_dim *. config.compression_ratio) in
          let features = zeros [n_samples; target_dim] ~device:config.device in
          
          (* Process in batches *)
          for i = 0 to (n_samples - 1) / config.batch_size do
            let start_idx = i * config.batch_size in
            let length = min config.batch_size (n_samples - start_idx) in
            let batch = {
              features = narrow dataset.features ~dim:0 ~start:start_idx ~length;
              labels = narrow dataset.labels ~dim:0 ~start:start_idx ~length;
            } in
            
            (* Process batch through all checkpoints *)
            let batch_grads = List.map (fun checkpoint ->
              Checkpoint.load_model model checkpoint;
              process_batch model batch checkpoint config.use_mixed_precision
            ) checkpoints in
            
            (* Combine gradients *)
            let combined = List.fold_left add (List.hd batch_grads) (List.tl batch_grads) in
            
            (* Project if needed *)
            let projected = 
              if config.compression_ratio < 1.0 then
                let proj = WhitenedFeatureDistance.random_projection 
                  (shape2_exn combined |> snd) 
                  target_dim in
                mm combined proj
              else
                combined in
            
            (* Store results *)
            copy_ 
              (narrow features ~dim:0 ~start:start_idx ~length)
              projected
          done;
          
          save features cache_path;
          features
        end
    | None ->
        (* Process without caching *)
        embed_gradients_nocache ~config model checkpoints dataset
end

module WhitenedFeatureDistance = struct
  type whitening_stats = {
    mean: Tensor.t;
    scale: Tensor.t;
    eigenvalues: Tensor.t;
    eigenvectors: Tensor.t;
  }

  let compute_whitening_stats features =
    let n_samples = shape2_exn features |> fst in
    let mean = mean features ~dim:[0] ~keepdim:true in
    let centered = sub features mean in
    
    (* Compute SVD *)
    let u, s, v = svd centered ~some:true in
    
    (* Regularize eigenvalues *)
    let eps = 1e-6 in
    let s_reg = add s (full (shape s) eps) in
    let scale = sqrt (div_scalar s_reg (float_of_int (n_samples - 1))) in
    
    { mean; scale; eigenvalues = s_reg; eigenvectors = v }

  let apply_whitening features stats =
    let centered = sub features stats.mean in
    let rotated = mm centered stats.eigenvectors in
    div rotated stats.scale

  let update_stats old_stats new_features alpha =
    let new_stats = compute_whitening_stats new_features in
    {
      mean = add 
        (mul_scalar old_stats.mean (1.0 -. alpha))
        (mul_scalar new_stats.mean alpha);
      scale = add
        (mul_scalar old_stats.scale (1.0 -. alpha))
        (mul_scalar new_stats.scale alpha);
      eigenvalues = add
        (mul_scalar old_stats.eigenvalues (1.0 -. alpha))
        (mul_scalar new_stats.eigenvalues alpha);
      eigenvectors = add
        (mul_scalar old_stats.eigenvectors (1.0 -. alpha))
        (mul_scalar new_stats.eigenvectors alpha);
    }

  let compute_wfd x y =
    let stats = compute_whitening_stats (Tensor.cat [x; y] ~dim:0) in
    let x_white = apply_whitening x stats in
    let y_white = apply_whitening y stats in
    Tensor.mse_divergence x_white y_white ~reduction:Mean
end

module OptimalTransport = struct
  type selection_stats = {
    ot_distances: float array;
    selected_indices: int list;
    computation_time: float;
    memory_peak: int64;
  }

  type sinkhorn_params = {
    epsilon: float;
    max_iter: int;
    tolerance: float;
    stabilize_freq: int;
  }

  let default_sinkhorn_params = {
    epsilon = 0.1;
    max_iter = 1000;
    tolerance = 1e-6;
    stabilize_freq = 10;
  }

  let sinkhorn ?(params=default_sinkhorn_params) cost_matrix =
    let n, m = shape2_exn cost_matrix in
    
    let log_mu = full [n] (-.log (float_of_int n)) in
    let log_nu = full [m] (-.log (float_of_int m)) in
    
    let rec iterate alpha beta k total_err =
      if k >= params.max_iter then (alpha, beta)
      else
        let log_k = div_scalar (neg cost_matrix) params.epsilon in
        let log_u = sub log_mu (logsumexp (add log_k beta) ~dim:[1] ~keepdim:false) in
        let log_v = sub log_nu (logsumexp (add (transpose log_k ~dim0:0 ~dim1:1) alpha) 
          ~dim:[1] ~keepdim:false) in
        
        let alpha_new = add alpha log_u in
        let beta_new = add beta log_v in
        let err_u = norm (sub alpha_new alpha) ~p:Float.infinity in
        let err_v = norm (sub beta_new beta) ~p:Float.infinity in
        let err = max (to_float0_exn err_u) (to_float0_exn err_v) in

        if err < params.tolerance then (alpha_new, beta_new)
        else iterate alpha_new beta_new (k + 1) err
    in
    
    let alpha_init = zeros [n] in
    let beta_init = zeros [m] in
    let alpha, beta = iterate alpha_init beta_init 0 Float.infinity in
    
    let pi = exp (div_scalar (neg cost_matrix) params.epsilon) in
    let coupling = mul (exp alpha) (mul pi (exp beta)) in
    
    { coupling; cost = to_float0_exn (sum (mul cost_matrix coupling)) }

  let compute_ot_distance source target ~epsilon =
    let cost_matrix = Tensor.zeros [Tensor.shape2_exn source.features |> fst; 
                                  Tensor.shape2_exn target.features |> fst] in
    Tensor.iteri (fun i s_feat ->
      Tensor.iteri (fun j t_feat ->
        let cost = WhitenedFeatureDistance.compute_wfd s_feat t_feat in
        Tensor.set cost_matrix [|i; j|] cost
      ) target.features
    ) source.features;
    
    let params = { default_sinkhorn_params with epsilon } in
    let plan = sinkhorn ~params cost_matrix in
    plan.cost

module KFoldSelection = struct
  type fold_config = {
    k: int;
    validation_metric: Tensor.t -> Tensor.t -> float;
    min_fold_size: int;
    shuffle: bool;
    seed: int option;
  }

  let default_config = {
    k = 10;
    validation_metric = (fun x y -> Tensor.mse_divergence x y ~reduction:Mean);
    min_fold_size = 100;
    shuffle = true;
    seed = Some 42;
  }

  let create_folds ?(config=default_config) dataset =
    let n_samples = Tensor.shape2_exn dataset.features |> fst in
    if n_samples < config.k * config.min_fold_size in
    let indices = Array.init n_samples (fun i -> i) in
    begin match config.seed with
    | Some seed -> Random.init seed
    | None -> ()
    end;
    
    if config.shuffle then
      Array.sort (fun _ _ -> if Random.bool () then 1 else -1) indices;
    
    let fold_size = n_samples / config.k in
    let extra = n_samples mod config.k in
    
    List.init config.k (fun k ->
      let start_idx = k * fold_size + min k extra in
      let length = fold_size + if k < extra then 1 else 0 in
      Array.sub indices start_idx length |> Array.to_list
    )

  let select_with_validation ?(config=default_config) candidate_data target_data =
    let folds = create_folds ~config target_data in
    let fold_results = List.mapi (fun fold_idx validation_indices ->
      let selection_indices = 
        List.filter (fun i -> not (List.mem i validation_indices))
          (List.init (Tensor.shape2_exn target_data.features |> fst) (fun i -> i)) in
      
      let selection_data = {
        features = Tensor.index_select target_data.features ~dim:0 
          ~index:(Tensor.of_int1 selection_indices);
        labels = Tensor.index_select target_data.labels ~dim:0 
          ~index:(Tensor.of_int1 selection_indices);
      } in
      
      let validation_data = {
        features = Tensor.index_select target_data.features ~dim:0 
          ~index:(Tensor.of_int1 validation_indices);
        labels = Tensor.index_select target_data.labels ~dim:0 
          ~index:(Tensor.of_int1 validation_indices);
      } in
      
      let selected_indices = Selection.select candidate_data selection_data in
      let validation_cost = OptimalTransport.compute_ot_distance
        (extract_samples candidate_data selected_indices)
        validation_data
        ~epsilon:0.1 in
        
      selected_indices, validation_cost
    ) folds in
    
    let all_indices = List.map fst fold_results |> List.flatten |> List.sort_uniq compare in
    let all_costs = List.map snd fold_results in
    
    all_indices,
    { OptimalTransport.
      ot_distances = Array.of_list all_costs;
      selected_indices = all_indices;
      computation_time = Unix.gettimeofday ();
      memory_peak = Gc.allocated_bytes () |> Int64.of_float;
    }

module Selection = struct
  type selection_mode = FixedSize | OptimalTransport

  type coupling_params = {
    transport_weight: float;
    coverage_weight: float;
    diversity_weight: float;
    temperature: float;
  }

  type stopping_criteria = {
    max_iterations: int;
    min_improvement: float;
    patience: int;
    max_selections: int;
  }

  type selection_params = {
    mode: selection_mode;
    initial_size: int;
    growth_factor: float;
    coupling: coupling_params;
    stopping: stopping_criteria;
    device: Device.t;
  }

  let default_params = {
    mode = OptimalTransport;
    initial_size = 100;
    growth_factor = 1.5;
    coupling = {
      transport_weight = 1.0;
      coverage_weight = 0.3;
      diversity_weight = 0.2;
      temperature = 0.1;
    };
    stopping = {
      max_iterations = 1000;
      min_improvement = 1e-4;
      patience = 5;
      max_selections = 10000;
    };
    device = Device.Cpu;
  }

  let compute_coupling_score selected_data target_data params =
    let transport_cost = OptimalTransport.compute_ot_distance 
      selected_data target_data ~epsilon:params.coupling.temperature in
    
    let coverage_score =
      let dist_matrix = Tensor.pairwise_distance 
        selected_data.features target_data.features in
      let min_dists = Tensor.min dist_matrix ~dim:[0] ~keepdim:false |> fst in
      Tensor.mean min_dists |> Tensor.to_float0_exn in
    
    let diversity_score =
      let dist_matrix = Tensor.pairwise_distance 
        selected_data.features selected_data.features in
      let min_dists = Tensor.min dist_matrix ~dim:[0] ~keepdim:false |> fst in
      Tensor.mean min_dists |> Tensor.to_float0_exn in
    
    transport_cost *. params.coupling.transport_weight -.
    coverage_score *. params.coupling.coverage_weight +.
    diversity_score *. params.coupling.diversity_weight

  let select ?(params=default_params) candidate_data target_data =
    let rec selection_loop selected_indices size iteration last_best_score patience_count =
      if iteration >= params.stopping.max_iterations 
         || List.length selected_indices >= params.stopping.max_selections 
         || patience_count >= params.stopping.patience then
        selected_indices
      else
        let candidates = 
          match params.mode with
          | FixedSize -> fixed_size_selection candidate_data target_data ~size
          | OptimalTransport -> 
              KFoldSelection.select_with_validation candidate_data target_data |> fst
        in
        
        let selected_data = extract_samples candidate_data candidates in
        let score = compute_coupling_score selected_data target_data params in
        
        let improvement = last_best_score -. score in
        if improvement < params.stopping.min_improvement then
          selection_loop candidates
            (int_of_float (float_of_int size *. params.growth_factor))
            (iteration + 1)
            score
            (patience_count + 1)
        else
          selection_loop candidates
            (int_of_float (float_of_int size *. params.growth_factor))
            (iteration + 1)
            score
            0
    in
    
    selection_loop [] params.initial_size 0 Float.infinity 0

module Training = struct
  type weighted_sampler = {
    indices: int array;
    weights: float array;
    alias: int array;
    prob: float array;
  }

  let create_weighted_sampler weights =
    let n = Array.length weights in
    let sum_weights = Array.fold_left (+.) 0. weights in
    let avg = sum_weights /. float_of_int n in
    
    let indices = Array.init n (fun i -> i) in
    let prob = Array.map (fun w -> w *. float_of_int n /. sum_weights) weights in
    let alias = Array.make n 0 in
    
    let small = Stack.create () in
    let large = Stack.create () in
    
    Array.iteri (fun i p ->
      if p < 1.0 then Stack.push i small
      else Stack.push i large
    ) prob;
    
    while not (Stack.is_empty small) && not (Stack.is_empty large) do
      let l = Stack.pop small in
      let g = Stack.pop large in
      alias.(l) <- g;
      prob.(g) <- prob.(g) +. prob.(l) -. 1.0;
      if prob.(g) < 1.0 then Stack.push g small
      else Stack.push g large
    done;
    
    while not (Stack.is_empty large) do
      let g = Stack.pop large in
      prob.(g) <- 1.0
    done;
    while not (Stack.is_empty small) do
      let l = Stack.pop small in
      prob.(l) <- 1.0
    done;
    
    { indices; weights; alias; prob }

  let create_weighted_dataset data weights =
    let sampler = create_weighted_sampler weights in
    let n_samples = Array.length weights in
    
    let sample_indices = Array.init n_samples (fun _ ->
      let i = Random.int n_samples in
      if Random.float 1.0 < sampler.prob.(i) then
        sampler.indices.(i)
      else
        sampler.indices.(sampler.alias.(i))
    ) in
    
    {
      features = Tensor.index_select data.features ~dim:0 
        ~index:(Tensor.of_int1 (Array.to_list sample_indices));
      labels = Tensor.index_select data.labels ~dim:0 
        ~index:(Tensor.of_int1 (Array.to_list sample_indices));
    }

  let train_weighted model optimizer data weights ~epochs ~batch_size =
    for epoch = 0 to epochs - 1 do
      let weighted_dataset = create_weighted_dataset data weights in
      let n_samples = Tensor.shape2_exn weighted_dataset.features |> fst in
      
      for batch_start = 0 to n_samples - 1 step batch_size do
        let batch_end = min (batch_start + batch_size) n_samples in
        let batch = {
          features = Tensor.narrow weighted_dataset.features ~dim:0 
            ~start:batch_start ~length:(batch_end - batch_start);
          labels = Tensor.narrow weighted_dataset.labels ~dim:0 
            ~start:batch_start ~length:(batch_end - batch_start);
        } in
        
        let predicted = Model.forward model batch.features in
        let divergence = Divergence.cross_entropy predicted batch.labels in
        
        Optimizer.zero_grad optimizer;
        Tensor.backward divergence;
        Optimizer.step optimizer;
      done;
    done