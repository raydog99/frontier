open Torch
open Price_volume_encoder
open Earnings_autoencoder

type price_volume_data = {
  price: Tensor.t;
  volume: Tensor.t;
}

type earnings_data = Tensor.t

type movement = Up | Down | Hold

type t = {
  price_volume_encoder: Price_volume_encoder.t;
  earnings_autoencoder: Earnings_autoencoder.t;
  transformer: Transformer.t;
  cpc: Cpc.t;
  classifier: Nn.t;
}

let create price_volume_dim earnings_dim hidden_dim num_heads =
  {
    price_volume_encoder = Price_volume_encoder.create price_volume_dim hidden_dim;
    earnings_autoencoder = Earnings_autoencoder.create earnings_dim hidden_dim;
    transformer = Transformer.create hidden_dim num_heads;
    cpc = Cpc.create hidden_dim;
    classifier = Nn.sequential [
      Nn.linear hidden_dim hidden_dim;
      Nn.relu ();
      Nn.linear hidden_dim 3; (* 3 classes: Up, Down, Hold *)
    ];
  }

let forward t price_volume earnings =
  let pv_encoded = Price_volume_encoder.forward t.price_volume_encoder (Tensor.cat [price_volume.price; price_volume.volume] ~dim:1) in
  let (e_encoded, _) = Earnings_autoencoder.forward t.earnings_autoencoder earnings in
  let combined = Tensor.(pv_encoded + e_encoded) in
  let transformed = Transformer.forward t.transformer combined in
  Nn.apply t.classifier transformed

let cpc_loss t c_t z_pos z_neg temperature =
  let (f_k_pos, f_k_neg) = Cpc.forward t.cpc c_t z_pos z_neg in
  Cpc.infonce_loss f_k_pos f_k_neg temperature

let classification_loss t price_volume earnings labels =
  let predictions = forward t price_volume earnings in
  Nn.cross_entropy_loss predictions ~targets:labels

let predict t price_volume earnings =
  let logits = forward t price_volume earnings in
  let probabilities = Tensor.softmax logits ~dim:1 in
  let predicted_class = Tensor.argmax probabilities ~dim:1 ~keepdim:false in
  (probabilities, predicted_class)

let save t ~filename =
  let state_dict = [
    ("price_volume_encoder", Price_volume_encoder.state_dict t.price_volume_encoder);
    ("earnings_autoencoder", Earnings_autoencoder.state_dict t.earnings_autoencoder);
    ("transformer", Transformer.state_dict t.transformer);
    ("cpc", Cpc.state_dict t.cpc);
    ("classifier", Nn.state_dict t.classifier);
  ] in
  Serialize.save_multi state_dict ~filename

let load t ~filename =
  let state_dict = Serialize.load_multi ~filename in
  Price_volume_encoder.load_state_dict t.price_volume_encoder (List.assoc "price_volume_encoder" state_dict);
  Earnings_autoencoder.load_state_dict t.earnings_autoencoder (List.assoc "earnings_autoencoder" state_dict);
  Transformer.load_state_dict t.transformer (List.assoc "transformer" state_dict);
  Cpc.load_state_dict t.cpc (List.assoc "cpc" state_dict);
  Nn.load_state_dict t.classifier (List.assoc "classifier" state_dict);
  t
end

let train model train_loader val_loader num_epochs learning_rate temperature =
let optimizer = Optimizer.adam (CET.parameters model) ~lr:learning_rate in
let lr_scheduler = Training_utils.LRScheduler.create optimizer learning_rate 0.95 5 in
let best_val_accuracy = ref 0. in
let patience = 10 in
let epochs_without_improvement = ref 0 in

for epoch = 1 to num_epochs do
  Training_utils.LRScheduler.step lr_scheduler epoch;
  let total_loss = ref 0. in
  let num_batches = ref 0 in
  Data_loader.iter train_loader ~f:(fun (price_volume, earnings, labels) ->
    let batch_size = Tensor.shape price_volume.price |> List.hd in
    let z_pos = CET.forward model price_volume earnings in
    let z_neg = Tensor.stack (List.init batch_size (fun _ -> Dataset.get_negative_sample train_loader.dataset 0 |> (fun (pv, e, _) -> CET.forward model pv e))) ~dim:0 in
    let loss = Training_utils.train_step model optimizer price_volume earnings z_pos z_neg labels temperature in
    total_loss := !total_loss +. Tensor.to_float0_exn loss;
    num_batches := !num_batches + 1
  );
  let avg_train_loss = !total_loss /. float_of_int !num_batches in
  let (val_loss, val_accuracy) = Training_utils.evaluate model val_loader in
  Printf.printf "Epoch %d, Train Loss: %.4f, Val Loss: %.4f, Val Accuracy: %.4f\n"
    epoch avg_train_loss val_loss val_accuracy;

  if val_accuracy > !best_val_accuracy then (
    best_val_accuracy := val_accuracy;
    epochs_without_improvement := 0;
    CET.save model ~filename:"best_model.ot";
  ) else (
    epochs_without_improvement := !epochs_without_improvement + 1;
  );

  if !epochs_without_improvement >= patience then (
    Printf.printf "Early stopping triggered. Best validation accuracy: %.4f\n" !best_val_accuracy;
    exit 0
  )