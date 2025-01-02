open Torch

type metrics = {
  mse: float;
  mae: float;
  relative_error: float;
  max_error: float;
}

type monitor_state = {
  epoch: int;
  metrics: metrics;
  best_model_state: Tensor.t list;
  early_stop_counter: int;
}

let compute_metrics prediction target =
  let mse = Tensor.mse_loss prediction target |> Tensor.float_value in
  let mae = Tensor.abs (prediction - target) 
            |> Tensor.mean |> Tensor.float_value in
  let rel_err = Tensor.div (Tensor.abs (prediction - target)) 
                          (Tensor.abs target)
               |> Tensor.mean |> Tensor.float_value in
  let max_err = Tensor.abs (prediction - target)
                |> Tensor.max |> fst |> Tensor.float_value in
  { mse; mae; relative_error = rel_err; max_error = max_err }

let create_monitor () = {
  epoch = 0;
  metrics = {
    mse = Float.infinity;
    mae = Float.infinity;
    relative_error = Float.infinity;
    max_error = Float.infinity;
  };
  best_model_state = [];
  early_stop_counter = 0;
}

let update monitor model data =
  let input, target = data in
  let prediction = WavKANNetwork.forward model input in
  let current_metrics = compute_metrics prediction target in
  
  (* Update best model state if needed *)
  let best_state, counter = 
    if current_metrics.mse < monitor.metrics.mse then
      (WavKANNetwork.parameters model, 0)
    else
      (monitor.best_model_state, monitor.early_stop_counter + 1)
  in
  
  {
    epoch = monitor.epoch + 1;
    metrics = current_metrics;
    best_model_state = best_state;
    early_stop_counter = counter;
  }

module Adaptive = struct
  open Torch

  type adaptive_params = {
    scale: Tensor.t;
    translation: Tensor.t;
    frequency: Tensor.t option;
    shape: Tensor.t option;
  }

  type strategy =
    | Gradient
    | MetaLearning
    | Evolutionary

  type t = {
    input_dim: int;
    output_dim: int;
    params: adaptive_params;
    strategy: strategy;
    mutable adaptation_history: adaptive_params list;
  }

  let create input_dim output_dim strategy =
    let params = {
      scale = Tensor.ones [output_dim] ~requires_grad:true;
      translation = Tensor.zeros [output_dim] ~requires_grad:true;
      frequency = Some (Tensor.ones [output_dim] ~requires_grad:true);
      shape = None;
    } in
    {
      input_dim;
      output_dim;
      params;
      strategy;
      adaptation_history = [];
    }

  let gradient_update params learning_rate =
    {
      scale = params.scale - 
              Tensor.mul_scalar (Option.get (Tensor.grad params.scale)) 
                               learning_rate;
      translation = params.translation - 
                   Tensor.mul_scalar (Option.get (Tensor.grad params.translation))
                                   learning_rate;
      frequency = Option.map (fun f ->
        f - Tensor.mul_scalar (Option.get (Tensor.grad f)) learning_rate
      ) params.frequency;
      shape = params.shape;
    }

  let meta_update params learning_rate =
    (* Meta-learning based parameter update *)
    let meta_step param grad =
      param - Tensor.mul_scalar (Option.get grad) learning_rate in
    {
      scale = meta_step params.scale (Tensor.grad params.scale);
      translation = meta_step params.translation (Tensor.grad params.translation);
      frequency = Option.map (fun f -> 
        meta_step f (Tensor.grad f)
      ) params.frequency;
      shape = params.shape;
    }

  let evolutionary_update params mutation_strength =
    (* Evolution strategy based update *)
    let mutate param =
      let noise = Tensor.randn (Tensor.shape param) in
      param + Tensor.mul_scalar noise mutation_strength in
    {
      scale = mutate params.scale;
      translation = mutate params.translation;
      frequency = Option.map mutate params.frequency;
      shape = params.shape;
    }

  let forward t input =
    (* Apply current adaptive parameters *)
    let scaled = Tensor.div input t.params.scale in
    let translated = Tensor.sub scaled t.params.translation in
    
    (* Apply frequency modulation if available *)
    let modulated = match t.params.frequency with
      | Some freq -> 
          let phase = Tensor.mul translated freq in
          Tensor.cos phase
      | None -> translated in
    
    (* Apply shape modification if available *)
    let output = match t.params.shape with
      | Some shape -> Tensor.mul modulated shape
      | None -> modulated in
    
    output

  let adapt t =
    let new_params = match t.strategy with
      | Gradient -> gradient_update t.params 0.01
      | MetaLearning -> meta_update t.params 0.01
      | Evolutionary -> evolutionary_update t.params 0.01
    in
    t.adaptation_history <- t.params :: t.adaptation_history;
    t.params <- new_params;
    t

  let get_adaptation_history t =
    t.params :: t.adaptation_history
end

module Training = struct
  open Torch

  type training_config = {
    batch_size: int;
    epochs: int;
    optimizer_config: Optimization.optimizer_config;
    regularizers: Regularization.regularizer_type list;
    early_stopping_patience: int;
  }

  let train_epoch model optimizer data_loader config =
    let total_loss = ref 0. in
    let n_batches = ref 0 in
    
    data_loader (fun input target ->
      Optimization.zero_grad optimizer;
      
      let output = WavKANNetwork.forward model input in
      let loss = Tensor.mse_loss output target in
      
      (* Add regularization *)
      let reg_loss = Regularization.compute_penalty 
        (WavKANNetwork.parameters model) config.regularizers in
      let total = Tensor.(loss + reg_loss) in
      
      Tensor.backward total;
      Optimization.step optimizer;
      
      total_loss := !total_loss +. Tensor.float_value total;
      incr n_batches
    );
    
    !total_loss /. float !n_batches

  let train model config data_loader validation_loader =
    let optimizer = Optimization.create config.optimizer_config model in
    let monitor = Validation.create_monitor () in
    
    let rec train_loop monitor epoch =
      if epoch >= config.epochs || 
         monitor.Validation.early_stop_counter >= config.early_stopping_patience
      then monitor
      else begin
        WavKANNetwork.train model true;
        let train_loss = train_epoch model optimizer data_loader config in
        
        WavKANNetwork.train model false;
        let new_monitor = Validation.update monitor model 
          (validation_loader ()) in
        
        train_loop new_monitor (epoch + 1)
      end
    in
    train_loop monitor 0
end