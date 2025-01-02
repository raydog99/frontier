open Torch

type dataset = {
  t: Tensor.t;
  x: Tensor.t;
  r: Tensor.t;
  v: Tensor.t;
}

type data_loader = unit -> (Tensor.t * Tensor.t * Tensor.t * Tensor.t) list

let create_dataset size =
  let t = Tensor.(rand [size; 1] ~kind:(T Float)) in
  let x = Tensor.(rand [size; 1] ~kind:(T Float) * float_vec [|10.|]) in
  let r = Tensor.(rand [size; 1] ~kind:(T Float) * float_vec [|5.|]) in
  let v = Tensor.(rand [size; 1] ~kind:(T Float)) in
  {t; x; r; v}

let split_dataset dataset split_ratio =
  let size = Tensor.shape dataset.t |> List.hd in
  let train_size = int_of_float (float_of_int size *. split_ratio) in
  let train_dataset = {
    t = Tensor.narrow dataset.t ~dim:0 ~start:0 ~length:train_size;
    x = Tensor.narrow dataset.x ~dim:0 ~start:0 ~length:train_size;
    r = Tensor.narrow dataset.r ~dim:0 ~start:0 ~length:train_size;
    v = Tensor.narrow dataset.v ~dim:0 ~start:0 ~length:train_size;
  } in
  let test_dataset = {
    t = Tensor.narrow dataset.t ~dim:0 ~start:train_size ~length:(size - train_size);
    x = Tensor.narrow dataset.x ~dim:0 ~start:train_size ~length:(size - train_size);
    r = Tensor.narrow dataset.r ~dim:0 ~start:train_size ~length:(size - train_size);
    v = Tensor.narrow dataset.v ~dim:0 ~start:train_size ~length:(size - train_size);
  } in
  (train_dataset, test_dataset)

let create_data_loader dataset batch_size =
  let size = Tensor.shape dataset.t |> List.hd in
  let num_batches = size / batch_size in
  fun () ->
    let idx = Tensor.randperm size ~kind:(T Int64) in
    let t = Tensor.index_select dataset.t ~dim:0 ~index:idx in
    let x = Tensor.index_select dataset.x ~dim:0 ~index:idx in
    let r = Tensor.index_select dataset.r ~dim:0 ~index:idx in
    let v = Tensor.index_select dataset.v ~dim:0 ~index:idx in
    List.init num_batches (fun i ->
      let start = i * batch_size in
      let t_batch = Tensor.narrow t ~dim:0 ~start ~length:batch_size in
      let x_batch = Tensor.narrow x ~dim:0 ~start ~length:batch_size in
      let r_batch = Tensor.narrow r ~dim:0 ~start ~length:batch_size in
      let v_batch = Tensor.narrow v ~dim:0 ~start ~length:batch_size in
      (t_batch, x_batch, r_batch, v_batch)
    )

let get_test_data data_loader =
  let batches = data_loader () in
  let (_, _, _, v) = List.hd batches in
  v