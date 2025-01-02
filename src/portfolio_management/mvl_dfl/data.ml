open Torch

type t = {
  features : Tensor.t;
  returns : Tensor.t;
}

let load_data filename =
  match Util.read_csv filename with
  | Error e -> Error e
  | Ok data ->
    let headers = List.hd data in
    let data_without_headers = List.tl data in
    let features, returns = 
      List.map (fun row ->
        let features = List.take 30 row in
        let returns = List.drop 30 row in
        features, returns
      ) data_without_headers
      |> List.split
    in
    match Util.to_tensor features, Util.to_tensor returns with
    | Ok features, Ok returns -> Ok { features; returns }
    | Error e, _ | _, Error e -> Error e

let preprocess data lookback_period =
  try
    let features = Tensor.unfold data.features ~dimension:0 ~size:lookback_period ~step:1 in
    let returns = Tensor.narrow data.returns ~dim:0 ~start:lookback_period ~length:(Tensor.shape data.returns |> List.hd - lookback_period) in
    Ok { features; returns }
  with
  | e -> Error (Printf.sprintf "Error preprocessing data: %s" (Printexc.to_string e))

let split data =
  match Util.split_data data 0.8 with
  | Error e -> Error e
  | Ok (train, val_test) ->
    match Util.split_data val_test 0.5 with
    | Error e -> Error e
    | Ok (val_, test) -> Ok (train, val_, test)

let get_batch data batch_size =
  try
    let data_size = Tensor.shape data.features |> List.hd in
    let indices = Tensor.randint ~high:data_size ~size:[batch_size] in
    let batch_features = Tensor.index_select data.features ~dim:0 ~index:indices in
    let batch_returns = Tensor.index_select data.returns ~dim:0 ~index:indices in
    Ok { features = batch_features; returns = batch_returns }
  with
  | e -> Error (Printf.sprintf "Error getting batch: %s" (Printexc.to_string e))

let parallel_preprocess data lookback_period num_workers =
  let open Lwt in
  let chunk_size = (Tensor.shape data.features |> List.hd) / num_workers in
  
  let process_chunk start_idx =
    let end_idx = min (start_idx + chunk_size) (Tensor.shape data.features |> List.hd) in
    let chunk_features = Tensor.narrow data.features ~dim:0 ~start:start_idx ~length:(end_idx - start_idx) in
    let chunk_returns = Tensor.narrow data.returns ~dim:0 ~start:start_idx ~length:(end_idx - start_idx) in
    
    Lwt_preemptive.detach (fun () ->
      let processed_features = Tensor.unfold chunk_features ~dimension:0 ~size:lookback_period ~step:1 in
      let processed_returns = Tensor.narrow chunk_returns ~dim:0 ~start:lookback_period ~length:(Tensor.shape chunk_returns |> List.hd - lookback_period) in
      (processed_features, processed_returns)
    ) ()
  in
  
  let chunks = List.init num_workers (fun i -> i * chunk_size) in
  Lwt_main.run (
    Lwt_list.map_p process_chunk chunks >>= fun results ->
    let processed_features = Tensor.cat (List.map fst results) ~dim:0 in
    let processed_returns = Tensor.cat (List.map snd results) ~dim:0 in
    Lwt.return { features = processed_features; returns = processed_returns }
  )

let bootstrap data =
  let size = Tensor.shape data.features |> List.hd in
  let indices = Tensor.randint ~high:size ~size:[size] in
  {
    features = Tensor.index_select data.features ~dim:0 ~index:indices;
    returns = Tensor.index_select data.returns ~dim:0 ~index:indices;
  }