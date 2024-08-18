open Torch
open Error_handling

type t = {
  embedding : nn;
  layer_norm : nn;
  attention_layers : nn list;
  output_layer : nn;
}

let create input_dim hidden_dim num_layers num_heads =
  let embedding = Layer.linear input_dim hidden_dim in
  let layer_norm = Layer.layer_norm [hidden_dim] in
  let attention_layers =
    List.init num_layers (fun _ ->
      Layer.multi_head_attention hidden_dim num_heads)
  in
  let output_layer = Layer.linear hidden_dim 3 in (* 3 classes: up, down, unchanged *)
  { embedding; layer_norm; attention_layers; output_layer }

let forward t input =
  let embedded = Layer.forward t.embedding input in
  let normalized = Layer.forward t.layer_norm embedded in
  let attended = List.fold_left (fun x layer -> Layer.forward layer x) normalized t.attention_layers in
  Layer.forward t.output_layer attended

let loss_fn predicted actual =
  Tensor.cross_entropy_loss predicted actual ~reduction:Mean

let train t dataset learning_rate num_epochs =
  try
    let optimizer = Optimizer.adam (Layer.parameters t.embedding) ~learning_rate in
    for epoch = 1 to num_epochs do
      let total_loss = ref 0. in
      Dataset.iter dataset ~f:(fun batch ->
        let { Dataset.Batch.inputs; targets } = batch in
        Optimizer.zero_grad optimizer;
        let predicted = forward t inputs in
        let loss = loss_fn predicted targets in
        total_loss := !total_loss +. Tensor.to_float0_exn loss;
        Tensor.backward loss;
        Optimizer.step optimizer;
      );
      info (Printf.sprintf "Epoch %d/%d, Loss: %.4f" epoch num_epochs (!total_loss /. float (Dataset.length dataset)));
    done
  with
  | _ -> raise_error "Failed to train the model"

let predict t input =
  let output = forward t input in
  Tensor.softmax output ~dim:1

let save_model t filename =
  let state_dict = List.flatten [
    Layer.state_dict t.embedding;
    Layer.state_dict t.layer_norm;
    List.flatten (List.map Layer.state_dict t.attention_layers);
    Layer.state_dict t.output_layer;
  ] in
  Serialize.save_multi ~named_tensors:state_dict ~filename

let load_model filename input_dim hidden_dim num_layers num_heads =
  let t = create input_dim hidden_dim num_layers num_heads in
  let state_dict = Serialize.load_multi ~filename in
  List.iter2 (fun param (name, value) -> Tensor.copy_ param value) 
    (List.flatten [
      Layer.parameters t.embedding;
      Layer.parameters t.layer_norm;
      List.flatten (List.map Layer.parameters t.attention_layers);
      Layer.parameters t.output_layer;
    ])
    state_dict;
  t