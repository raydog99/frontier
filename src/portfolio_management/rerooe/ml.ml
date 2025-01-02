open Torch

type model =
  | LSTM of { num_layers: int; hidden_size: int; dropout: float }
  | Transformer of { num_layers: int; num_heads: int; d_model: int; dropout: float }
  | RandomForest of { num_trees: int; max_depth: int }
  | EnsembleModel of model list

let create_model model_type input_size output_size =
  match model_type with
  | LSTM { num_layers; hidden_size; dropout } ->
      Layer.lstm ~num_layers ~input_size ~hidden_size ~dropout ()
  | Transformer { num_layers; num_heads; d_model; dropout } ->
      Layer.transformer ~num_layers ~num_heads ~d_model ~dropout ()
  | RandomForest { num_trees; max_depth } ->
      let dummy_data = List.init 1000 (fun _ -> 
        (Array.init input_size (fun _ -> Random.float 1.), Random.float 1.)
      ) in
      Random_forest.create num_trees max_depth dummy_data
  | EnsembleModel models ->
      (* Create an ensemble of models *)
      let sub_models = List.map (fun m -> create_model m input_size output_size) models in
      Layer.sequential (
        sub_models @ 
        [Layer.linear ~[Layer.linear ~in_features:(List.length models * output_size) ~out_features:output_size ()]
      )

let train model data_loader num_epochs learning_rate =
  let optimizer = Optimizer.adam model.parameters ~learning_rate in
  for epoch = 1 to num_epochs do
    Torch_utils.Dataloader.iter data_loader ~f:(fun batch ->
      let { Torch_utils.Dataloader.inputs; targets } = batch in
      Optimizer.zero_grad optimizer;
      let predicted = Layer.forward model inputs in
      let loss = mse_loss predicted targets in
      backward loss;
      Optimizer.step optimizer;
    )
  done

let predict model input =
  no_grad (fun () ->
    Layer.forward model input
  )

let feature_importance model =
  match model with
  | RandomForest rf -> Random_forest.feature_importance rf
  | _ ->
      failwith "Feature importance not available for this model type"

let cross_validate model data_loader num_folds =
  let fold_size = Torch_utils.Dataloader.length data_loader / num_folds in
  let fold_results = ref [] in
  for i = 0 to num_folds - 1 do
    let train_loader, val_loader = Torch_utils.Dataloader.split data_loader ~ratio:(float_of_int i /. float_of_int num_folds) in
    train model train_loader 100 0.001;
    let val_loss = 
      Torch_utils.Dataloader.fold val_loader ~init:0. ~f:(fun acc batch ->
        let { Torch_utils.Dataloader.inputs; targets } = batch in
        let predicted = predict model inputs in
        let loss = mse_loss predicted targets in
        acc +. Tensor.to_float0_exn loss
      ) /. float_of_int fold_size
    in
    fold_results := val_loss :: !fold_results
  done;
  List.rev !fold_results