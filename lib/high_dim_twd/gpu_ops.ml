open Torch

let batch_pairwise_distances points1 points2 config =
  let device_points1 = Config.get_device_tensor config points1 in
  let device_points2 = Config.get_device_tensor config points2 in
  
  let distances = 
    if config.use_mixed_precision then
      let p1 = Tensor.to_type device_points1 ~dtype:Half in
      let p2 = Tensor.to_type device_points2 ~dtype:Half in
      let dist = Tensor.cdist p1 p2 in
      Tensor.to_type dist ~dtype:Float
    else
      Tensor.cdist device_points1 device_points2
  in
  
  match config.device with
  | CPU -> distances
  | GPU _ -> Tensor.cpu distances

let parallel_sparse_mm sp1 sp2 config =
  (* Move sparse matrices to GPU if available *)
  let to_gpu_sparse sparse = 
    match config.device with
    | CPU -> sparse
    | GPU device_id -> 
        {sparse with 
         Sparse_ops.indices = Tensor.cuda sparse.indices ~device_id;
         Sparse_ops.values = Tensor.cuda sparse.values ~device_id}
  in
  
  let gpu_sp1 = to_gpu_sparse sp1 in
  let gpu_sp2 = to_gpu_sparse sp2 in
  
  let n_chunks = match config.device with
    | CPU -> 1
    | GPU _ -> Cuda.device_count ()
  in
  
  if n_chunks = 1 then
    Sparse_ops.sparse_mm gpu_sp1

let parallel_sparse_mm sp1 sp2 config =
  (* Move sparse matrices to GPU if available *)
  let to_gpu_sparse sparse = 
    match config.device with
    | CPU -> sparse
    | GPU device_id -> 
        {sparse with 
         Sparse_ops.indices = Tensor.cuda sparse.indices ~device_id;
         Sparse_ops.values = Tensor.cuda sparse.values ~device_id}
  in
  
  let gpu_sp1 = to_gpu_sparse sp1 in
  let gpu_sp2 = to_gpu_sparse sp2 in
  
  let n_chunks = match config.device with
    | CPU -> 1
    | GPU _ -> Cuda.device_count ()
  in
  
  if n_chunks = 1 then
    Sparse_ops.sparse_mm gpu_sp1 gpu_sp2
  else
    (* Distribute computation across GPUs *)
    let chunk_size = Tensor.size gpu_sp1.values 0 / n_chunks in
    let results = Array.init n_chunks (fun i ->
      let start_idx = i * chunk_size in
      let length = if i = n_chunks - 1 
        then Tensor.size gpu_sp1.values 0 - start_idx 
        else chunk_size in
      
      let chunk_indices = Tensor.narrow gpu_sp1.indices ~dim:1 ~start:start_idx ~length in
      let chunk_values = Tensor.narrow gpu_sp1.values ~dim:0 ~start:start_idx ~length in
      let chunk = Sparse_ops.create_sparse chunk_indices chunk_values gpu_sp1.size in
      
      Domain.spawn (fun () ->
        let device_id = i mod Cuda.device_count () in
        let config = {config with device = GPU device_id} in
        Sparse_ops.sparse_mm chunk (to_gpu_sparse gpu_sp2)
      )
    ) in
    
    (* Combine results *)
    let combined = Array.map Domain.join results 
      |> Array.fold_left (fun acc chunk ->
        let combined_indices = Tensor.cat [acc.indices; chunk.indices] ~dim:1 in
        let combined_values = Tensor.cat [acc.values; chunk.values] ~dim:0 in
        Sparse_ops.create_sparse combined_indices combined_values acc.size
      ) (Array.get results 0 |> Domain.join)
    in
    
    (* Move result back to CPU if needed *)
    match config.device with
    | CPU -> combined
    | GPU _ -> 
        {combined with 
         indices = Tensor.cpu combined.indices;
         values = Tensor.cpu combined.values}