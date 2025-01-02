open Torch
open Autoencoder
open Translator
open Logging

type t = {
  ae1: Autoencoder.t;
  ae2: Autoencoder.t;
  translator1: Translator.t;
  translator2: Translator.t;
}

let create input_channels =
  let ae1 = Autoencoder.create input_channels in
  let ae2 = Autoencoder.create input_channels in
  let translator1 = Translator.create 10 10 20 in
  let translator2 = Translator.create 10 10 20 in
  { ae1; ae2; translator1; translator2 }

let kl_divergence q p =
  let log_q = Tensor.log q in
  let log_p = Tensor.log p in
  Tensor.mean (Tensor.(q * (log_q - log_p)))

let train_step t input learning_rate is_ae1_speaking =
  if is_ae1_speaking then
    (* AE1 speaks, AE2 listens *)
    let loss1, code1 = Autoencoder.train t.ae1 input learning_rate in
    let translated1 = Translator.forward t.translator1 code1 in
    let loss2, code2 = Autoencoder.train t.ae2 input learning_rate in
    let kl_loss = kl_divergence code2 translated1 in
    (loss1, loss2, kl_loss)
  else
    (* AE2 speaks, AE1 listens *)
    let loss2, code2 = Autoencoder.train t.ae2 input learning_rate in
    let translated2 = Translator.forward t.translator2 code2 in
    let loss1, code1 = Autoencoder.train t.ae1 input learning_rate in
    let kl_loss = kl_divergence code1 translated2 in
    (loss1, loss2, kl_loss)

let train t data num_epochs learning_rate =
  let num_samples = Tensor.shape_dim data 0 in
  for epoch = 1 to num_epochs do
    let total_loss1 = ref 0. in
    let total_loss2 = ref 0. in
    let total_kl_loss = ref 0. in
    for i = 0 to num_samples - 1 do
      let input = Tensor.slice data ~dim:0 ~start:i ~end_:(i+1) in
      let is_ae1_speaking = i mod 2 = 0 in
      let (loss1, loss2, kl_loss) = train_step t input learning_rate is_ae1_speaking in
      total_loss1 := !total_loss1 +. Tensor.float_value loss1;
      total_loss2 := !total_loss2 +. Tensor.float_value loss2;
      total_kl_loss := !total_kl_loss +. Tensor.float_value kl_loss;
    done;
    Logging.info (Printf.sprintf "Epoch %d: AE1 Loss: %.4f, AE2 Loss: %.4f, KL Loss: %.4f"
      epoch (!total_loss1 /. float num_samples) (!total_loss2 /. float num_samples) (!total_kl_loss /. float num_samples));
  done

let extract_patterns t data k =
  let _, codes1 = Autoencoder.forward t.ae1 data in
  let _, codes2 = Autoencoder.forward t.ae2 data in
  let combined_codes = Tensor.cat [codes1; codes2] ~dim:1 in
  combined_codes