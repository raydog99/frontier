open Torch

let chunk_size = 1000

let build_affinity_matrix_streaming points epsilon =
  let n = Tensor.size points 0 in
  let num_chunks = (n + chunk_size - 1) / chunk_size in
  let sparse_chunks = Array.make num_chunks (Sparse_ops.create_sparse 
    (Tensor.zeros [|2; 0|]) (Tensor.zeros [|0|]) [|chunk_size; n|]) in
  
  for i = 0 to num_chunks - 1 do
    let start_idx = i * chunk_size in
    let end_idx = min ((i + 1) * chunk_size) n in
    let chunk = Tensor.narrow points ~dim:0 ~start:start_idx ~length:(end_idx - start_idx) in
    
    let chunk_dists = Tensor.cdist chunk points in
    let chunk_affinity = Tensor.exp (Tensor.div_scalar 
      (Tensor.neg (Tensor.mul chunk_dists chunk_dists)) epsilon) in
    
    sparse_chunks.(i) <- Sparse_ops.from_dense chunk_affinity 1e-10
  done;
  
  sparse_chunks

let normalize_affinity_streaming sparse_chunks =
  let total_size = Array.fold_left (fun acc chunk -> 
    acc + Tensor.size chunk.Sparse_ops.values 0) 0 sparse_chunks in
  
  (* Compute row sums in streaming fashion *)
  let row_sums = Tensor.zeros [|Array.get (Array.get sparse_chunks 0).Sparse_ops.size 0|] in
  Array.iter (fun chunk ->
    let dense = Sparse_ops.to_dense chunk in
    let chunk_sums = Tensor.sum dense ~dim:[1] in
    Tensor.add_ row_sums chunk_sums
  ) sparse_chunks;
  
  (* Normalize chunks *)
  Array.map (fun chunk ->
    let indices = chunk.Sparse_ops.indices in
    let values = chunk.Sparse_ops.values in
    let n = Tensor.size values 0 in
    
    let normalized_values = Tensor.zeros [|n|] in
    for i = 0 to n - 1 do
      let row = Tensor.get indices [|0; i|] |> Int.of_float in
      let value = Tensor.get values [|i|] in
      let row_sum = Tensor.get row_sums [|row|] in
      Tensor.set normalized_values [|i|] (value /. Float.sqrt(row_sum +. 1e-10))
    done;
    
    Sparse_ops.create_sparse indices normalized_values chunk.Sparse_ops.size
  ) sparse_chunks |> Array.to_list