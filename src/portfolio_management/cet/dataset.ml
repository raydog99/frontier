open Torch
open Cet
open Dat_preprocessing

type t = {
  price_volume: Cet.price_volume_data;
  earnings: Cet.earnings_data;
  labels: Tensor.t;
  sequence_length: int;
}

let create price volume earnings labels sequence_length =
  let price_volume = preprocess_price_volume price volume in
  let earnings = preprocess_earnings earnings in
  { price_volume; earnings; labels; sequence_length }

let length t =
  Tensor.shape t.price_volume.price |> List.hd |> (fun x -> x - t.sequence_length + 1)

let get t idx =
  let slice_tensor tensor =
    Tensor.narrow tensor ~dim:0 ~start:idx ~length:t.sequence_length
  in
  let price_volume = {
    price = slice_tensor t.price_volume.price |> Data_augmentation.augment_tensor;
    volume = slice_tensor t.price_volume.volume |> Data_augmentation.augment_tensor;
  } in
  let earnings = slice_tensor t.earnings in
  let label = Tensor.get t.labels [idx + t.sequence_length - 1] in
  (price_volume, earnings, label)

let get_negative_sample t idx =
  let random_idx = Random.int (length t) in
  get t random_idx