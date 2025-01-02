open Torch

let process_large_dataset ~samples ~batch_size ~f =
  let n = Tensor.size samples 0 in
  let num_batches = (n + batch_size - 1) / batch_size in
  
  let results = Array.init num_batches (fun i ->
    let start = i * batch_size in
    let length = min batch_size (n - start) in
    let batch = Tensor.narrow samples ~dim:0 ~start ~length in
    f batch
  ) in
  
  Array.fold_left Tensor.add results.(0) 
    (Array.sub results 1 (Array.length results - 1))

let efficient_kronecker ~samples ~batch_size =
  let n, d = Tensor.size samples 0, Tensor.size samples 1 in
  
  let process_batch batch =
    let batch_size = Tensor.size batch 0 in
    Array.init batch_size (fun i ->
      let sample = Tensor.slice batch ~dim:0 ~start:i ~end_:(i+1) in
      let flat = Tensor.reshape sample [|-1|] in
      Kronecker_ops.efficient_outer_product flat flat
    ) |> Tensor.stack ~dim:0 in
  
  process_large_dataset ~samples ~batch_size ~f:process_batch

let memory_efficient_multiply ~a ~b ~memory_limit =
  let block_size = Fast_matrix_ops.optimal_block_size 
    (Tensor.size a 1) memory_limit in
  Fast_matrix_ops.block_multiply a b block_size