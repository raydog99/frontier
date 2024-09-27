open Torch
open Dataset

type t = {
  dataset: Dataset.t;
  batch_size: int;
  shuffle: bool;
}

let create dataset batch_size shuffle =
  { dataset; batch_size; shuffle }

let iter t ~f =
  let indices = List.init (Dataset.length t.dataset) (fun i -> i) in
  let indices = if t.shuffle then List.sort (fun _ _ -> Random.int 3 - 1) indices else indices in
  List.iter (fun batch_indices ->
    let batch = List.map (Dataset.get t.dataset) batch_indices in
    let price_volume_batch, earnings_batch, labels_batch = List.split3 batch in
    let price_volume_batch = {
      Cet.price = Tensor.stack (List.map (fun pv -> pv.Cet.price) price_volume_batch) ~dim:0;
      volume = Tensor.stack (List.map (fun pv -> pv.Cet.volume) price_volume_batch) ~dim:0;
    } in
    let earnings_batch = Tensor.stack earnings_batch ~dim:0 in
    let labels_batch = Tensor.stack labels_batch ~dim:0 in
    f (price_volume_batch, earnings_batch, labels_batch)
  ) (List.groupi indices ~break:(fun i _ _ -> i mod t.batch_size = 0))