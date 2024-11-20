open Torch

module Types = struct
  type distribution =
    | Categorical of { probs: Tensor.t; num_classes: int }
    | Normal of { mu: Tensor.t; sigma: Tensor.t }
    | Bernoulli of { probs: Tensor.t }
    
  type feature_type =
    | Continuous
    | Categorical of int
    | Binary
    
  type feature_info = {
    feature_type: feature_type;
    name: string;
    index: int;
    dependencies: int list;
  }
  
  type model_config = {
    feature_dim: int;
    hidden_dims: int list;
    num_classes: int;
    feature_info: feature_info array;
    dropout_rate: float;
    use_batch_norm: bool;
    residual_connections: bool;
  }
end

module Utils = struct
  let kl_divergence p q =
    Tensor.(sum (p * (log p - log q)) ~dim:[1])
    
  let entropy x =
    Tensor.(mean (- x * log x) ~dim:[1])
    
  let normalize tensor =
    let mean = Tensor.mean tensor ~dim:[0] ~keepdim:true in
    let std = Tensor.std tensor ~dim:[0] ~keepdim:true ~unbiased:true in
    Tensor.((tensor - mean) / std)
    
  let one_hot tensor num_classes =
    Tensor.(zeros [size tensor 0; num_classes])
    |> Tensor.scatter_ ~dim:1 ~src:(ones_like tensor) ~index:tensor
    
  let sample_gumbel_like tensor =
    let u = Tensor.uniform_like tensor ~low:0. ~high:1. in
    Tensor.(log (- log u))
end

module Dataset = struct
  type t = {
    features: Tensor.t;
    labels: Tensor.t;
    batch_size: int;
  }
  
  let create features labels batch_size =
    { features; labels; batch_size }
    
  let shuffle t =
    let n = Tensor.size t.features 0 in
    let idx = Tensor.(randperm n) in
    {
      features = Tensor.index_select t.features ~dim:0 ~index:idx;
      labels = Tensor.index_select t.labels ~dim:0 ~index:idx;
      batch_size = t.batch_size;
    }
    
  let batches t =
    let n = Tensor.size t.features 0 in
    let num_batches = (n + t.batch_size - 1) / t.batch_size in
    List.init num_batches (fun i ->
      let start_idx = i * t.batch_size in
      let end_idx = min (start_idx + t.batch_size) n in
      let batch_features = Tensor.narrow t.features ~dim:0 ~start:start_idx ~len:(end_idx - start_idx) in
      let batch_labels = Tensor.narrow t.labels ~dim:0 ~start:start_idx ~len:(end_idx - start_idx) in
      (batch_features, batch_labels)
    )
    
  let split t ratio =
    let n = Tensor.size t.features 0 in
    let train_size = int_of_float (float_of_int n *. ratio) in
    let train_features = Tensor.narrow t.features ~dim:0 ~start:0 ~len:train_size in
    let train_labels = Tensor.narrow t.labels ~dim:0 ~start:0 ~len:train_size in
    let val_features = Tensor.narrow t.features ~dim:0 ~start:train_size ~len:(n - train_size) in
    let val_labels = Tensor.narrow t.labels ~dim:0 ~start:train_size ~len:(n - train_size) in
    
    { features = train_features; labels = train_labels; batch_size = t.batch_size },
    { features = val_features; labels = val_labels; batch_size = t.batch_size }
end

module PolicyNetwork = struct  
  type layer = {
    linear: Layer.t;
    batch_norm: Layer.t option;
    dropout: float;
  }
  
  type t = {
    layers: layer list;
    dependency_encoder: Layer.t;
    feature_embeddings: Layer.t;
    scorer: Layer.t;
    config: model_config;
  }
  
  let create_layer in_dim out_dim config =
    {
      linear = Layer.linear ~in_dim ~out_dim ();
      batch_norm = if config.use_batch_norm then 
        Some (Layer.batch_norm1d out_dim) else None;
      dropout = config.dropout_rate;
    }
    
  let create config =
    let layers = List.mapi (fun i dim ->
      let in_dim = if i = 0 then config.feature_dim * 2 
                   else List.nth config.hidden_dims (i-1) in
      create_layer in_dim dim config
    ) config.hidden_dims in
    
    let last_dim = List.hd (List.rev config.hidden_dims) in
    {
      layers;
      dependency_encoder = Layer.linear ~in_dim:config.feature_dim 
                                      ~out_dim:config.feature_dim ();
      feature_embeddings = Layer.linear ~in_dim:config.feature_dim 
                                      ~out_dim:config.feature_dim ();
      scorer = Layer.linear ~in_dim:last_dim 
                          ~out_dim:config.feature_dim ();
      config;
    }
    
  let forward_layer layer input training =
    
    let x = Layer.forward layer.linear input in
    let x = match layer.batch_norm with
      | Some bn -> Layer.forward bn x ~training
      | None -> x in
    let x = relu x in
    if training && layer.dropout > 0. then
      dropout x layer.dropout ~training
    else x
    
  let forward t features mask training =
    (* Create dependency matrix from feature info *)
    let dep_matrix = Tensor.zeros [t.config.feature_dim; t.config.feature_dim] in
    Array.iter (fun info ->
      List.iter (fun dep ->
        Tensor.set dep_matrix [info.index; dep] (Tensor.float 1.)
      ) info.dependencies
    ) t.config.feature_info;
    
    (* Encode feature dependencies *)
    let dep_encoded = Layer.forward t.dependency_encoder 
      (matmul dep_matrix (features * mask)) in
      
    (* Create feature embeddings *)
    let feat_embedded = Layer.forward t.feature_embeddings features in
    
    (* Combine feature and dependency information *)
    let combined = cat [feat_embedded; dep_encoded] ~dim:1 in
    
    (* Process through layers *)
    let hidden = List.fold_left (fun x layer ->
      forward_layer layer x training
    ) combined t.layers in
    
    (* Generate selection probabilities *)
    let scores = Layer.forward t.scorer hidden in
    
    (* Mask out already selected features *)
    let masked_scores = scores * (ones_like mask - mask) +
                       (Tensor.full_like scores Float.neg_infinity) * mask in
    
    softmax masked_scores ~dim:(-1)
    
  let parameters t =
    let layer_params = List.concat_map (fun layer ->
      let params = Layer.parameters layer.linear in
      match layer.batch_norm with
      | Some bn -> params @ Layer.parameters bn
      | None -> params
    ) t.layers in
    layer_params @
    Layer.parameters t.dependency_encoder @
    Layer.parameters t.feature_embeddings @
    Layer.parameters t.scorer
end

module PredictorNetwork = struct
  type residual_block = {
    conv1: Layer.t;
    conv2: Layer.t;
    batch_norm1: Layer.t option;
    batch_norm2: Layer.t option;
    dropout: float;
  }
  
  type t = {
    input_projection: Layer.t;
    residual_blocks: residual_block list;
    classifier: Layer.t;
    config: model_config;
  }
  
  let create_residual_block dim config =
    {
      conv1 = Layer.linear ~in_dim:dim ~out_dim:dim ();
      conv2 = Layer.linear ~in_dim:dim ~out_dim:dim ();
      batch_norm1 = if config.use_batch_norm then 
        Some (Layer.batch_norm1d dim) else None;
      batch_norm2 = if config.use_batch_norm then 
        Some (Layer.batch_norm1d dim) else None;
      dropout = config.dropout_rate;
    }
    
  let create config =
    let num_blocks = 3 in
    let hidden_dim = List.hd config.hidden_dims in
    {
      input_projection = Layer.linear ~in_dim:config.feature_dim 
                                    ~out_dim:hidden_dim ();
      residual_blocks = List.init num_blocks (fun _ -> 
        create_residual_block hidden_dim config);
      classifier = Layer.linear ~in_dim:hidden_dim 
                              ~out_dim:config.num_classes ();
      config;
    }
    
  let process_features features config =
    
    Array.mapi (fun i info ->
      let feat = select features ~dim:1 ~index:(tensor [i]) in
      match info.feature_type with
      | Continuous -> Utils.normalize feat
      | Categorical n -> Utils.one_hot feat n
      | Binary -> feat
    ) config.feature_info
    |> Array.to_list
    |> cat ~dim:1
    
  let forward_residual_block block input training =
    let identity = input in
    let x = Layer.forward block.conv1 input in
    let x = match block.batch_norm1 with
      | Some bn -> Layer.forward bn x ~training
      | None -> x in
    let x = relu x in
    let x = if training && block.dropout > 0. then
      dropout x block.dropout ~training
    else x in
    
    let x = Layer.forward block.conv2 x in
    let x = match block.batch_norm2 with
      | Some bn -> Layer.forward bn x ~training
      | None -> x in
      
    relu (x + identity)
    
  let forward t features mask training =
    let processed_features = process_features features t.config in
    let x = Layer.forward t.input_projection (processed_features * mask) in
    
    let hidden = List.fold_left (fun x block ->
      forward_residual_block block x training
    ) x t.residual_blocks in
    
    Layer.forward t.classifier hidden
    
  let loss t features mask labels =
    let predictions = forward t features mask true in
    Tensor.cross_entropy_loss predictions labels
    
  let parameters t =
    let block_params = List.concat_map (fun block ->
      let params = Layer.parameters block.conv1 @ Layer.parameters block.conv2 in
      match block.batch_norm1, block.batch_norm2 with
      | Some bn1, Some bn2 -> params @ Layer.parameters bn1 @ Layer.parameters bn2
      | Some bn1, None -> params @ Layer.parameters bn1
      | None, Some bn2 -> params @ Layer.parameters bn2
      | None, None -> params
    ) t.residual_blocks in
    Layer.parameters t.input_projection @
    block_params @
    Layer.parameters t.classifier
end

module CMI = struct
  let estimate features response mask k =
    let batch_size = size features 0 in
    let feature_dim = size features 1 in
    
    (* Sample similar examples based on current mask *)
    let find_similar_examples features mask =
      let masked_features = features * mask in
      let distances = cdist masked_features masked_features in
      let _, indices = topk distances ~k ~dim:1 ~largest:false in
      indices in
      
    let similar_indices = find_similar_examples features mask k in
    
    (* Calculate conditional distributions *)
    let conditional_responses = Stack.init k (fun i ->
      let indices = similar_indices.%[i] in
      index_select response ~dim:0 ~index:indices
    ) |> stack ~dim:0 in
    
    (* Calculate KL divergence *)
    let mean_response = mean conditional_responses ~dim:0 in
    let kl_divs = Stack.init k (fun i ->
      Utils.kl_divergence conditional_responses.%[i] mean_response
    ) |> stack ~dim:0 in
    
    mean kl_divs ~dim:0 |> float_value
    
  module Oracle = struct
    type t = {
      feature_conditionals: Tensor.t;
      response_conditionals: Tensor.t;
    }
    
    let create features response =
      let feature_dim = Tensor.size features 1 in
      let num_classes = Tensor.size response 1 in
      
      (* Estimate empirical distributions *)
      let feature_conditionals = 
        Tensor.zeros [feature_dim; feature_dim] in
      let response_conditionals =
        Tensor.zeros [feature_dim; num_classes] in
        
      (* Fill distributions using empirical counts/averages *)
      for i = 0 to feature_dim - 1 do
        for j = 0 to feature_dim - 1 do
          let cond_prob = Tensor.(mean (features.%[i] * features.%[j])) in
          Tensor.set feature_conditionals [i; j] cond_prob
        done;
        
        let resp_probs = Tensor.(mean (features.%[i] * response)) in
        Tensor.copy_ (Tensor.narrow response_conditionals ~dim:0 ~start:i ~len:1) resp_probs
      done;
      
      { feature_conditionals; response_conditionals }
      
    let estimate_cmi t features idx =
      
      let p_x = index_select t.feature_conditionals ~dim:0 ~index:(tensor [idx]) in
      let p_y = index_select t.response_conditionals ~dim:0 ~index:(tensor [idx]) in
      
      let joint = p_x * p_y in
      let marginal_y = sum p_y ~dim:1 ~keepdim:true in
      let marginal_x = sum p_x ~dim:1 ~keepdim:true in
      
      Utils.kl_divergence joint (marginal_x * marginal_y) |> float_value
  end
  
  module GreedyPolicy = struct
    type t = {
      num_samples: int;
      oracle: Oracle.t option;
    }
    
    let create ?(num_samples=100) ?oracle () =
      { num_samples; oracle }
      
    let select_feature t features response current_mask =
      
      
      let feature_dim = size features 1 in
      let available_features = 
        ones [feature_dim] - sum current_mask ~dim:0 in
      
      (* Calculate CMI for each available feature *)
      let cmi_scores = Stack.init feature_dim (fun i ->
        if Tensor.get available_features [i] |> float_value < 0.5 then
          Float.neg_infinity
        else match t.oracle with
          | Some oracle -> Oracle.estimate_cmi oracle features i
          | None -> estimate features response current_mask t.num_samples
      ) |> stack ~dim:0 in
      
      (* Return one-hot vector for best feature *)
      let best_idx = argmax cmi_scores ~dim:0 ~keepdim:false in
      Utils.one_hot best_idx feature_dim
  end
end

module Training = struct
  module LRScheduler = struct
    type scheduler_type =
      | StepLR of { step_size: int; gamma: float }
      | CosineAnnealingLR of { T_max: int; eta_min: float }
      | ReduceLROnPlateau of {
          factor: float;
          patience: int;
          min_lr: float;
          mutable best_score: float;
          mutable counter: int;
        }
        
    type t = {
      mutable current_lr: float;
      scheduler_type: scheduler_type;
      mutable epoch: int;
    }
    
    let create ?(scheduler_type=StepLR { step_size=30; gamma=0.1 }) initial_lr =
      { current_lr = initial_lr; scheduler_type; epoch = 0 }
      
    let step t ?score () =
      t.epoch <- t.epoch + 1;
      match t.scheduler_type with
      | StepLR { step_size; gamma } ->
          if t.epoch mod step_size = 0 then
            t.current_lr <- t.current_lr *. gamma
      | CosineAnnealingLR { T_max; eta_min } ->
          let progress = float_of_int (t.epoch mod T_max) /. float_of_int T_max in
          t.current_lr <- eta_min +. 0.5 *. (1. +. cos (Float.pi *. progress))
      | ReduceLROnPlateau p ->
          Option.iter (fun score ->
            if score > p.best_score then (
              p.best_score <- score;
              p.counter <- 0
            ) else (
              p.counter <- p.counter + 1;
              if p.counter >= p.patience then (
                t.current_lr <- max p.min_lr (t.current_lr *. p.factor);
                p.counter <- 0
              )
            )
          ) score
  end
  
  module Temperature = struct
    type t = {
      mutable current: float;
      min_temp: float;
      decay_rate: float;
    }
    
    let create ?(initial=1.0) ?(min_temp=0.1) ?(decay_rate=0.95) () =
      { current = initial; min_temp; decay_rate }
      
    let step t =
      t.current <- max t.min_temp (t.current *. t.decay_rate)
  end
  
  module Validation = struct
    type t = {
      mutable best_score: float;
      mutable best_model_state: (string * Tensor.t) list;
      patience: int;
      mutable counter: int;
      mutable should_stop: bool;
    }
    
    let create ?(patience=10) () =
      { 
        best_score = Float.neg_infinity;
        best_model_state = [];
        patience;
        counter = 0;
        should_stop = false;
      }
      
    let step t model score =
      if score > t.best_score then (
        t.best_score <- score;
        t.best_model_state <- model#state_dict;
        t.counter <- 0
      ) else (
        t.counter <- t.counter + 1;
        if t.counter >= t.patience then
          t.should_stop <- true
      )
      
    let restore_best_model t model =
      model#load_state_dict t.best_model_state
  end
  
  let train_epoch model optimizer dataset temperature =
    let total_loss = ref 0. in
    let num_batches = ref 0 in
    
    List.iter (fun (features, labels) ->
      let batch_size = size features 0 in
      let feature_dim = size features 1 in
      
      (* Initialize mask *)
      let current_mask = zeros [batch_size; feature_dim] in
      
      (* Zero gradients *)
      optimizer#zero_grad;
      
      (* Accumulate loss through selection steps *)
      let step_losses = ref [] in
      let final_mask = ref current_mask in
      
      for _ = 1 to model.config.num_selections do
        let step_loss, new_mask = DFS.step model features !final_mask labels temperature.current in
        step_losses := step_loss :: !step_losses;
        final_mask := new_mask
      done;
      
      (* Compute total loss and backpropagate *)
      let total_step_loss = 
        List.fold_left (fun acc loss -> acc + loss) 
          (zeros []) !step_losses in
      
      total_step_loss#backward;
      optimizer#step;
      
      total_loss := !total_loss +. float_value total_step_loss;
      num_batches := !num_batches + 1
    ) (Dataset.batches dataset);
    
    !total_loss /. float_of_int !num_batches
end

module Metrics = struct
  type evaluation_metrics = {
    accuracy: float;
    mi_score: float;
    selection_stability: float;
    avg_num_features: float;
  }
  
  let accuracy predictions labels =
    
    let predicted = argmax predictions ~dim:1 ~keepdim:false in
    let correct = eq predicted labels in
    mean correct ~dtype:Float |> float_value
    
  let mutual_information features labels mask =
    CMI.estimate features labels mask 100
    
  let selection_stability masks =
    let pairwise_overlap = matmul masks (transpose masks ~dim0:0 ~dim1:1) in
    mean pairwise_overlap |> float_value
    
  let feature_importance features labels =
    (* Calculate mutual information for each feature *)
    let feature_dim = size features 1 in
    let importances = Stack.init feature_dim (fun i ->
      let mask = Utils.one_hot (tensor [i]) feature_dim in
      mutual_information features labels mask
    ) |> stack ~dim:0 in
    
    normalize importances
end

module DFS = struct  
  type t = {
    policy: PolicyNetwork.t;
    predictor: PredictorNetwork.t;
    config: model_config;
  }
  
  let create config =
    {
      policy = PolicyNetwork.create config;
      predictor = PredictorNetwork.create config;
      config;
    }
    
  let select_feature t features current_mask temperature =
    let probs = PolicyNetwork.forward t.policy features current_mask true in
    
    if temperature > 0. then
      (* Gumbel-Softmax sampling *)
      let gumbel = Utils.sample_gumbel_like probs in
      let logits = Tensor.((log probs + gumbel) / (float temperature)) in
      Tensor.softmax logits ~dim:(-1)
    else
      (* Deterministic selection *)
      let idx = Tensor.argmax probs ~dim:1 ~keepdim:false in
      Utils.one_hot idx (t.config.feature_dim)
      
  let predict t features mask =
    PredictorNetwork.forward t.predictor features mask false
    
  let step t features current_mask labels temperature =
    let new_mask = select_feature t features current_mask temperature in
    let updated_mask = Tensor.(current_mask + new_mask) in
    let pred_loss = PredictorNetwork.loss t.predictor features updated_mask labels in
    pred_loss, updated_mask
    
  let train t train_dataset valid_dataset ~num_epochs ~initial_lr =
    let scheduler = Training.LRScheduler.create initial_lr in
    let validation = Training.Validation.create () in
    let temperature = Training.Temperature.create () in
    
    let optimizer = Optimizer.adam (PolicyNetwork.parameters t.policy @ 
                                  PredictorNetwork.parameters t.predictor)
                                 ~lr:scheduler.current_lr in
    
    for epoch = 1 to num_epochs do
      let train_loss = Training.train_epoch t optimizer train_dataset temperature in
      let metrics = evaluate t valid_dataset in
      
      Training.LRScheduler.step scheduler ~score:metrics.accuracy ();
      Training.Validation.step validation t metrics.accuracy;
      Training.Temperature.step temperature;
      
      if validation.should_stop then break
    done;
    
    Training.Validation.restore_best_model validation t
    
  let evaluate t dataset =
    let accuracies = ref [] in
    let mi_scores = ref [] in
    let selected_masks = ref [] in
    
    List.iter (fun (features, labels) ->
      let mask = zeros [size features 0; t.config.feature_dim] in
      let final_mask = ref mask in
      
      for _ = 1 to t.config.num_selections do
        let _, new_mask = step t features !final_mask labels 0. in
        final_mask := new_mask
      done;
      
      let predictions = predict t features !final_mask in
      
      accuracies := accuracy predictions labels :: !accuracies;
      mi_scores := mutual_information features labels !final_mask :: !mi_scores;
      selected_masks := !final_mask :: !selected_masks
    ) (Dataset.batches dataset);
    
    {
      accuracy = average !accuracies;
      mi_score = average !mi_scores;
      selection_stability = Metrics.selection_stability (stack !selected_masks ~dim:0);
      avg_num_features = float_of_int t.config.num_selections;
    }
end