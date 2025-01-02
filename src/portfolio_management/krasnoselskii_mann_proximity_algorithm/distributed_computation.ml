open Torch

let split_workload devices data =
  let num_devices = List.length devices in
  let data_size = Tensor.shape data |> List.hd in
  let chunk_size = data_size / num_devices in
  List.mapi (fun i device ->
    let start = i * chunk_size in
    let end_ = if i = num_devices - 1 then data_size else (i + 1) * chunk_size in
    (device, Tensor.narrow data 0 start (end_ - start))
  ) devices

let gather_results results =
  Tensor.cat results 0