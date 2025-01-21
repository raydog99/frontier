open Torch

(* Core types for mixture models *)
type observation = Tensor.t
type parameters = Tensor.t
type mixture_indicator = int
type network_params = {
  input_dim: int;
  hidden_dims: int list;
  output_dim: int;
  learning_rate: float;
}

(* Base Neural Network *)
module CoreNetwork = struct
  type t = {
    layers: nn;
    optimizer: Optimizer.t;
  }

  let create input_dim hidden_dims output_dim =
    let layers = 
      let rec build_layers dims acc =
        match dims with
        | [] -> acc
        | d::rest ->
          let linear = Layer.linear ~in_dim:input_dim ~out_dim:d () in
          let activation = Tensor.relu in
          build_layers rest (linear :: activation :: acc)
      in
      build_layers hidden_dims [] in
    let final_layer = Layer.linear ~in_dim:(List.hd (List.rev hidden_dims)) ~out_dim:output_dim () in
    let all_layers = layers @ [final_layer] in
    let optimizer = Optimizer.adam ~learning_rate:0.001 all_layers in
    { layers = all_layers; optimizer }

  let forward t x =
    List.fold_left (fun acc layer -> layer acc) x t.layers

  let summarize t x =
    let batch_size = (Tensor.shape x).(0) in
    let summary = forward t x in
    Tensor.reshape summary ~shape:[batch_size; -1]
  
  let create_classifier ~input_dim ~hidden_dims ~num_components =
    let network = create input_dim hidden_dims num_components in
    let optimizer = Optimizer.adam ~learning_rate:0.001 network.layers in
    { network with optimizer }

  let forward_classification t x theta =
    let combined = Tensor.cat [x; theta] ~dim:1 in
    let logits = forward t combined in
    Tensor.softmax logits ~dim:(-1)

  let categorical_cross_entropy logits labels =
    let log_probs = Tensor.log_softmax logits ~dim:(-1) in
    let target = Tensor.one_hot labels ~num_classes:(Tensor.shape logits).(1) in
    Tensor.mean (Tensor.sum (Tensor.mul (neg log_probs) target) ~dim:[1])
end

(* Distribution *)
module Distribution = struct
  type t = {
    loc: Tensor.t;
    scale: Tensor.t;
  }

  let gaussian ~loc ~scale = {
    loc;
    scale;
  }

  let sample dist ~n = 
    let eps = Tensor.randn [n] in
    Tensor.(dist.loc + (dist.scale * eps))

  let log_prob dist x =
    let open Tensor in
    let two_pi = Scalar.float 6.28318530718 in
    let variance = square dist.scale in
    let diff = x - dist.loc in
    neg (
      (log (sqrt (two_pi * variance))) +
      ((square diff) / (Scalar.float 2. * variance))
    )
end

(* Numerical Stability *)
module NumericalUtils = struct
  type stability_config = {
    eps: float;
    max_value: float;
    min_value: float;
    log_space_threshold: float;
  }

  let default_config = {
    eps = 1e-7;
    max_value = 1e7;
    min_value = 1e-7;
    log_space_threshold = 20.0;
  }

  let stabilize_log_sum_exp tensor ~dim config =
    let open Tensor in
    let max_val = max tensor ~dim ~keepdim:true |> fst in
    let shifted = sub tensor max_val in
    let sum_exp = sum (exp shifted) ~dim ~keepdim:true in
    add (log sum_exp) max_val

  let safe_log tensor config =
    let open Tensor in
    log (add tensor (float config.eps))

  let clip_gradients grads config =
    List.map (fun grad ->
      Tensor.(clamp grad 
        ~min:(float config.min_value) 
        ~max:(float config.max_value))
    ) grads
end

(* Memory Management *)
module MemoryManager = struct
  type buffer_config = {
    max_buffer_size: int;
    cleanup_threshold: float;
  }

  type t = {
    active_tensors: (string, Tensor.t) Hashtbl.t;
    config: buffer_config;
    total_memory: int ref;
  }

  let create config =
    { active_tensors = Hashtbl.create 16;
      config;
      total_memory = ref 0 }

  let register_tensor t name tensor =
    let size = Tensor.numel tensor in
    if !t.total_memory + size > t.config.max_buffer_size then
      Hashtbl.filter_map_inplace (fun _ tensor ->
        if Tensor.numel tensor < size then None
        else Some tensor
      ) t.active_tensors;
    Hashtbl.add t.active_tensors name tensor;
    t.total_memory := !t.total_memory + size

  let get_tensor t name =
    Hashtbl.find_opt t.active_tensors name

  let cleanup t =
    Hashtbl.clear t.active_tensors;
    t.total_memory := 0
end

(* Inference Network *)
module InferenceNetwork = struct
  type inference_t = {
    base: t;
    flow_layers: nn list;
  }

  let create_flow input_dim hidden_dims latent_dim =
    let base = create input_dim hidden_dims (2 * latent_dim) in
    let flow_layers = [
      Layer.linear ~in_dim:latent_dim ~out_dim:latent_dim ();
      Layer.tanh;
      Layer.linear ~in_dim:latent_dim ~out_dim:latent_dim ();
    ] in
    { base; flow_layers }

  let forward t x =
    let base_out = forward t.base x in
    let mu, log_var = Tensor.split ~dim:1 base_out in
    let std = Tensor.exp (Scalar.float 0.5 * log_var) in
    Distribution.gaussian ~loc:mu ~scale:std
end

(* Normalizing Flow *)
module NormalizingFlow = struct
  type layer = {
    weight: Tensor.t;
    bias: Tensor.t;
    scale: Tensor.t;
  }

  type t = {
    layers: layer list;
    dim: int;
  }

  let create dim num_layers =
    let create_layer () = {
      weight = Tensor.randn [dim; dim];
      bias = Tensor.randn [dim];
      scale = Tensor.randn [dim];
    } in
    { layers = List.init num_layers (fun _ -> create_layer ());
      dim }

  let forward t x =
    let log_det = ref (Tensor.zeros []) in
    let result = List.fold_left (fun acc layer ->
      let z = Tensor.(mm acc layer.weight + layer.bias) in
      let scale = Tensor.exp layer.scale in
      log_det := Tensor.(!log_det + sum (log scale) ~dim:[1]);
      Tensor.(z * scale)
    ) x t.layers in
    result, !log_det

  let inverse t x =
    List.fold_right (fun layer acc ->
      let scale = Tensor.exp layer.scale in
      let z = Tensor.(acc / scale) in
      Tensor.(mm z (transpose layer.weight ~dim0:0 ~dim1:1) - layer.bias)
    ) t.layers x
end

(* Forward Model *)
module ForwardModel = struct
  type t = {
    num_components: int;
    component_dim: int;
  }

  let create ~num_components ~component_dim = 
    { num_components; component_dim }

  let sample_prior t ~batch_size =
    let parameters = 
      Tensor.randn [batch_size; t.num_components * t.component_dim] in
    let indicators =
      Array.init batch_size (fun _ -> 
        Random.int t.num_components) in
    { parameters; indicators }

  let generate t state =
    let batch_size = (Tensor.shape state.parameters).(0) in
    let components = Array.init t.num_components (fun k ->
      let start_idx = k * t.component_dim in
      let end_idx = (k + 1) * t.component_dim in
      Tensor.narrow state.parameters ~dim:1 ~start:start_idx ~length:t.component_dim
    ) in
    Array.mapi (fun i indicator ->
      let component = components.(indicator) in
      let loc = Tensor.slice component ~dim:0 ~start:(Some i) ~length:(Some 1) in
      let dist = Distribution.gaussian 
        ~loc 
        ~scale:(Tensor.ones_like loc) in
      Distribution.sample dist ~n:1
    ) state.indicators
    |> Array.to_list
    |> Tensor.stack ~dim:0
end

(* Mixture Dependencies *)
module MixtureDependencies = struct
  type dependency_type =
    | Independent
    | Markov
    | SemiMarkov of { duration_max: int }
    | FullyDependent

  type transition_model =
    | Categorical of Tensor.t
    | Neural of CoreNetwork.t
    | Duration of {
        duration_net: CoreNetwork.t;
        transition_net: CoreNetwork.t;
      }

  type t = {
    dependency_type: dependency_type;
    num_components: int;
    transition_model: transition_model;
  }

  let create_transition_model dep_type num_components =
    match dep_type with
    | Independent -> 
        Categorical (Tensor.eye num_components)
    | Markov ->
        let net = CoreNetwork.create 
          num_components 
          [num_components * 2] 
          num_components in
        Neural net
    | SemiMarkov { duration_max } ->
        let duration_net = CoreNetwork.create 
          num_components 
          [num_components * 2] 
          duration_max in
        let transition_net = CoreNetwork.create 
          (num_components + duration_max) 
          [num_components * 2] 
          num_components in
        Duration { duration_net; transition_net }
    | FullyDependent ->
        let net = CoreNetwork.create 
          (num_components * 2) 
          [num_components * 4] 
          num_components in
        Neural net

  let sample_next_state t current_state history =
    match t.transition_model with
    | Categorical matrix ->
        let probs = Tensor.select matrix ~dim:0 ~index:current_state in
        let next_state = Tensor.multinomial probs ~num_samples:1 ~replacement:true in
        Tensor.int_value next_state
    | Neural net ->
        let input = Tensor.cat [
          Tensor.one_hot (Tensor.of_int1 [|current_state|]) ~num_classes:t.num_components;
          history
        ] ~dim:1 in
        let logits = CoreNetwork.forward net input in
        let probs = Tensor.softmax logits ~dim:(-1) in
        let next_state = Tensor.multinomial probs ~num_samples:1 ~replacement:true in
        Tensor.int_value next_state
    | Duration { duration_net; transition_net } ->
        let state_enc = Tensor.one_hot 
          (Tensor.of_int1 [|current_state|]) 
          ~num_classes:t.num_components in
        let duration_logits = CoreNetwork.forward duration_net state_enc in
        let duration = Tensor.multinomial 
          (Tensor.softmax duration_logits ~dim:(-1)) 
          ~num_samples:1 
          ~replacement:true in
        let combined = Tensor.cat [state_enc; duration] ~dim:1 in
        let trans_logits = CoreNetwork.forward transition_net combined in
        let next_state = Tensor.multinomial 
          (Tensor.softmax trans_logits ~dim:(-1)) 
          ~num_samples:1 
          ~replacement:true in
        Tensor.int_value next_state
end

(* Advanced Sampling *)
module AdvancedSampling = struct
  type sampling_config = {
    num_chains: int;
    warmup_steps: int;
    adaptation_steps: int;
    target_acceptance: float;
  }

  module HMC = struct
    type state = {
      position: Tensor.t;
      momentum: Tensor.t;
      log_prob: float;
      grad: Tensor.t;
    }

    let leapfrog state ~step_size ~num_steps ~potential_fn ~grad_fn =
      let rec integrate state = function
        | 0 -> state
        | n -> 
            let momentum = Tensor.(state.momentum + 
              (mul_scalar state.grad (Scalar.float (step_size /. 2.)))) in
            let position = Tensor.(state.position + 
              (mul_scalar momentum (Scalar.float step_size))) in
            let log_prob, grad = grad_fn position in
            let momentum = Tensor.(momentum + 
              (mul_scalar grad (Scalar.float (step_size /. 2.)))) in
            integrate { position; momentum; log_prob; grad } (n - 1)
      in
      integrate state num_steps

    let sample ~init_state ~config ~potential_fn ~grad_fn =
      let rec sampling_loop state chain step =
        if step >= config.warmup_steps + config.num_chains then
          List.rev chain
        else
          let momentum = Tensor.randn_like state.position in
          let init_energy = Tensor.(
            (sum (square momentum) |> float_value) /. 2. -. state.log_prob
          ) in
          let step_size = if step < config.adaptation_steps 
            then 0.1 else 0.01 in
          let proposed = leapfrog 
            { state with momentum }
            ~step_size 
            ~num_steps:10 
            ~potential_fn 
            ~grad_fn in
          let final_energy = Tensor.(
            (sum (square proposed.momentum) |> float_value) /. 2. -. 
            proposed.log_prob
          ) in
          let accept_prob = exp (init_energy -. final_energy) in
          let new_state = 
            if Random.float 1.0 < accept_prob then proposed else state in
          let new_chain = 
            if step >= config.warmup_steps 
            then new_state.position :: chain 
            else chain in
          sampling_loop new_state new_chain (step + 1)
      in
      sampling_loop init_state [] 0
  end

  module BifurcatingSampler = struct
    type tree = {
      position: Tensor.t;
      momentum: Tensor.t;
      grad: Tensor.t;
      log_prob: float;
      depth: int;
    }

    let build_tree state ~step_size ~potential_fn ~grad_fn ~max_depth =
      let rec build depth state =
        if depth = 0 then
          let proposed = HMC.leapfrog state 
            ~step_size 
            ~num_steps:1 
            ~potential_fn 
            ~grad_fn in
          proposed, proposed
        else
          let left, right = build (depth - 1) state in
          let direction = if Random.bool () then 1 else -1 in
          let (next_state, new_right) = 
            if direction = 1 then
              let r, _ = build (depth - 1) right in
              r, r
            else
              let l, _ = build (depth - 1) left in
              l, left in
          next_state, new_right
      in
      build max_depth state
  end
end

(* Training Configuration and Pipeline *)
module Training = struct
  type training_config = {
    batch_size: int;
    num_epochs: int;
    learning_rate: float;
    weight_decay: float;
    gradient_clip_norm: float option;
    early_stopping_patience: int option;
    learning_rate_schedule: [`Step | `Exponential | `Cosine] option;
  }

  type training_state = {
    epoch: int;
    divergence_history: float list;
    best_divergence: float;
    patience_counter: int;
    current_lr: float;
  }

  let create_scheduler config init_lr = match config with
    | `Step -> fun epoch ->
        if epoch mod 30 = 0 then init_lr *. 0.1 else init_lr
    | `Exponential -> fun epoch ->
        init_lr *. (0.95 ** float_of_int epoch)
    | `Cosine -> fun epoch ->
        let max_epochs = float_of_int 100 in
        let curr = float_of_int epoch in
        init_lr *. 0.5 *. (1. +. cos (Float.pi *. curr /. max_epochs))

  let train_model model config data =
    let state = ref {
      epoch = 0;
      divergence_history = [];
      best_divergence = Float.infinity;
      patience_counter = 0;
      current_lr = config.learning_rate;
    } in
    let scheduler = Option.map create_scheduler config.learning_rate_schedule in
    
    let rec training_loop () =
      if !state.epoch >= config.num_epochs then !state
      else match config.early_stopping_patience with
        | Some patience when !state.patience_counter > patience -> !state
        | _ ->
            let current_lr = match scheduler with
              | Some sched -> sched !state.epoch
              | None -> config.learning_rate in
            
            let total_divergence = ref 0. in
            let num_batches = (Tensor.shape data).(0) / config.batch_size in
            
            for batch = 0 to num_batches - 1 do
              let start_idx = batch * config.batch_size in
              let batch_data = Tensor.narrow data ~dim:0 
                ~start:start_idx 
                ~length:config.batch_size in
              
              let divergence = ABMM.train_step model batch_data in
              
              Option.iter (fun norm ->
                Parameters.clip_gradients model.inference_net.base.layers norm
              ) config.gradient_clip_norm;
              
              total_divergence := !total_divergence +. divergence;
            done;
            
            let avg_divergence = !total_divergence /. float_of_int num_batches in
            state := update_state !state avg_divergence;
            
            if !state.epoch mod 10 = 0 then
              Printf.printf "Epoch %d: Loss = %f\n" !state.epoch avg_divergence;
            
            training_loop ()
    in
    training_loop ()
end

(* Model Validation *)
module ModelValidation = struct
  type validation_metrics = {
    train_divergence: float;
    validation_divergence: float;
    classification_accuracy: float;
    posterior_predictive_score: float;
  }

  let compute_metrics model train_data validation_data =
    let train_divergence = ABMM.train_step model train_data in
    let validation_divergence = ABMM.train_step model validation_data in
    
    let params, class_probs = ABMM.infer model validation_data in
    let accuracy = 
      let pred_labels = Tensor.argmax class_probs ~dim:1 ~keepdim:false in
      let true_labels = Tensor.zeros [Tensor.shape validation_data |> Array.get 0] in
      Tensor.mean (Tensor.eq pred_labels true_labels)
      |> Tensor.float_value in
    
    let posterior_pred = 
      let samples = ForwardModel.generate model.forward_model params in
      let log_prob = Distribution.log_prob 
        (Distribution.gaussian 
           ~loc:samples 
           ~scale:(Tensor.ones_like samples)) 
        validation_data
      |> Tensor.mean
      |> Tensor.float_value in
    
    { train_divergence;
      validation_divergence;
      classification_accuracy = accuracy;
      posterior_predictive_score = log_prob }
end

(* Model Diagnostics *)
module ModelDiagnostics = struct
  type convergence_metrics = {
    parameter_gradients: Tensor.t;
    parameter_updates: Tensor.t;
    elbo_values: float array;
    kl_divergences: float array;
  }

  let compute_gradient_stats network =
    let params = network.CoreNetwork.layers in
    let grads = List.filter_map (fun layer ->
      match layer with
      | { Tensor.grad = Some grad; _ } -> Some grad
      | _ -> None
    ) params in
    let grad_norm = List.fold_left (fun acc grad ->
      Tensor.(acc + sum (square grad))
    ) (Tensor.zeros []) grads in
    Tensor.sqrt grad_norm

  type stability_metrics = {
    condition_number: float;
    gradient_norm: float;
    parameter_norm: float;
    numerical_error_estimate: float;
  }

  let monitor_numerical_stability model =
    let param_norm = compute_gradient_stats model.inference_net.base in
    let grad_norm = 
      List.map (fun layer ->
        match layer with
        | { Tensor.grad = Some grad; _ } -> 
            Tensor.(sum (abs grad) |> float_value)
        | _ -> 0.
      ) model.inference_net.base.layers
      |> List.fold_left max 0. in
    
    let condition_estimate = 
      let weights = List.filter_map (fun layer ->
        match layer with
        | { Tensor.requires_grad = true; _ } as w -> Some w
        | _ -> None
      ) model.inference_net.base.layers in
      match weights with
      | [] -> 1.
      | w::_ -> 
          let s = Tensor.svd w |> fun (_, s, _) -> s in
          let max_s = Tensor.max s ~dim:0 ~keepdim:true |> fst |> Tensor.float_value in
          let min_s = Tensor.min s ~dim:0 ~keepdim:true |> fst |> Tensor.float_value in
          max_s /. (min_s +. 1e-7) in
    
    { condition_number = condition_estimate;
      gradient_norm = grad_norm;
      parameter_norm = Tensor.float_value param_norm;
      numerical_error_estimate = grad_norm *. condition_estimate }
end

(* Stability Mechanisms *)
module StabilityMechanisms = struct
  type stability_mode =
    | Standard
    | LogSpace
    | ClippedGradients
    | AllStabilization

  let stabilize_tensor t mode =
    let open Tensor in
    match mode with
    | Standard -> t
    | LogSpace -> 
        let sign = sign t in
        mul sign (log1p (abs t))
    | ClippedGradients ->
        clamp t 
          ~min:(Scalar.float (-1e7)) 
          ~max:(Scalar.float 1e7)
    | AllStabilization ->
        t |> stabilize_tensor LogSpace
          |> stabilize_tensor ClippedGradients

  let safe_backward divergence mode =
    match mode with
    | Standard -> 
        Tensor.backward divergence
    | _ ->
        let stable_divergence = stabilize_tensor divergence mode in
        Tensor.backward stable_divergence;
        List.iter (fun grad ->
          match grad with
          | Some g -> stabilize_tensor g mode |> ignore
          | None -> ()
        ) (Tensor.grad_list divergence)
end

(* ABMM *)
module ABMM = struct
  type t = {
    forward_model: ForwardModel.t;
    inference_net: InferenceNetwork.inference_t;
    summary_net: CoreNetwork.t;
    classifier_net: CoreNetwork.t;
    sequential_net: CoreNetwork.t option;
    stability_mode: StabilityMechanisms.stability_mode;
    memory_manager: MemoryManager.t;
    optimizers: Optimizer.t list;
  }

  let create ~num_components ~component_dim ~summary_dim ~hidden_dims =
    let forward_model = ForwardModel.create ~num_components ~component_dim in
    let inference_net = InferenceNetwork.create_flow 
      summary_dim hidden_dims (num_components * component_dim) in
    let summary_net = CoreNetwork.create 
      component_dim hidden_dims summary_dim in
    let classifier_net = CoreNetwork.create_classifier
      ~input_dim:(component_dim + num_components * component_dim)
      ~hidden_dims ~num_components in
    
    { forward_model;
      inference_net;
      summary_net;
      classifier_net;
      sequential_net = None;
      stability_mode = StabilityMechanisms.Standard;
      memory_manager = MemoryManager.create 
        { max_buffer_size = 1000000; cleanup_threshold = 0.8 };
      optimizers = [
        inference_net.base.optimizer;
        summary_net.optimizer;
        classifier_net.optimizer;
      ] }

  let train_step t data =
    let open Tensor in
    let state = ForwardModel.sample_prior t.forward_model 
      ~batch_size:(shape data).(0) in
    let synthetic_data = ForwardModel.generate t.forward_model state in
    
    let summary = CoreNetwork.summarize t.summary_net data in
    let posterior = InferenceNetwork.forward t.inference_net summary in
    let samples = Distribution.sample posterior ~n:(shape data).(0) in
    
    let class_probs = CoreNetwork.forward_classification 
      t.classifier_net data samples in
    let class_divergence = CoreNetwork.categorical_cross_entropy 
      class_probs (Tensor.of_int1 state.indicators) in
    
    let kl_divergence = neg (mean (sum (Distribution.log_prob posterior samples) 
      ~dim:[1])) in
    
    let total_divergence = add class_divergence kl_divergence in
    
    StabilityMechanisms.safe_backward total_divergence t.stability_mode;
    List.iter Optimizer.step t.optimizers;
    
    float_value total_divergence

  let infer t data =
    let summary = CoreNetwork.summarize t.summary_net data in
    let posterior = InferenceNetwork.forward t.inference_net summary in
    let samples = Distribution.sample posterior ~n:(shape data).(0) in
    let class_probs = CoreNetwork.forward_classification 
      t.classifier_net data samples in
    posterior, class_probs
end

(* Integration *)
module Integration = struct
  type model_config = {
    num_components: int;
    input_dim: int;
    hidden_dims: int list;
    latent_dim: int;
    learning_rate: float;
    batch_size: int;
  }

  let create_and_train_model config data =
    let model = ABMM.create
      ~num_components:config.num_components
      ~component_dim:config.input_dim
      ~summary_dim:config.latent_dim
      ~hidden_dims:config.hidden_dims in
    
    let training_config = Training.{
      num_epochs = 100;
      batch_size = config.batch_size;
      learning_rate = config.learning_rate;
      weight_decay = 1e-5;
      gradient_clip_norm = Some 1.0;
      early_stopping_patience = Some 10;
      learning_rate_schedule = Some `Cosine;
    } in
    
    let final_state = Training.train_model model training_config data in
    model, final_state
end