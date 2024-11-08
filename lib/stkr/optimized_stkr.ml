open Torch
open Types
open Linalg

type optimized_model = {
  base_model: Stkr_core.model;
  kernel_cache: Kernel_cache.t;
  chunk_size: int;
  use_parallel: bool;
}

(* Create optimized model *)
let create base_model cache_size chunk_size use_parallel = {
  base_model;
  kernel_cache = Kernel_cache.create cache_size;
  chunk_size;
  use_parallel;
}

(* Parallel matrix multiplication helper *)
let parallel_matmul a b chunk_size =
  let m, n = Tensor.size a 0, Tensor.size b 1 in
  let num_threads = 4 in
  let chunk_per_thread = (m + num_threads - 1) / num_threads in
  
  let results = Array.make num_threads None in
  let threads = Array.init num_threads (fun thread_id ->
    Thread.create (fun () ->
      let start_row = thread_id * chunk_per_thread in
      let end_row = min (start_row + chunk_per_thread) m in
      if start_row < end_row then
        let chunk_result = Tensor_ops.chunked_matmul 
          (Tensor.narrow a 0 start_row (end_row - start_row))
          b chunk_size in
        results.(thread_id) <- Some chunk_result
    ) ()
  ) in
  
  Array.iter Thread.join threads;
  
  let result = Tensor.zeros [m; n] in
  Array.iteri (fun i chunk_opt ->
    match chunk_opt with
    | Some chunk ->
        let start_row = i * chunk_per_thread in
        let chunk_rows = Tensor.size chunk 0 in
        let dest = Tensor.narrow result 0 start_row chunk_rows in
        Tensor.copy_ dest chunk
    | None -> ()
  ) results;
  result

(* Optimized fit implementation *)
let fit model x_labeled y_labeled x_unlabeled =
  let n_labeled = Tensor.size x_labeled 0 in
  let x_all = Tensor.cat [x_labeled; x_unlabeled] 0 in
  
  (* Compute kernel matrix efficiently *)
  let compute_kernel_cached x y =
    let key = Kernel_cache.compute_key x y in
    Kernel_cache.get_or_compute model.kernel_cache key 
      model.base_model.kernel x y
  in
  
  let k_matrix = 
    if model.use_parallel then
      parallel_matmul 
        (compute_kernel_cached x_labeled x_all)
        (Tensor.transpose (compute_kernel_cached x_labeled x_all) 0 1)
        model.chunk_size
    else
      Tensor_ops.chunked_matmul 
        (compute_kernel_cached x_labeled x_all)
        (Tensor.transpose (compute_kernel_cached x_labeled x_all) 0 1)
        model.chunk_size
  in
  
  (* Transform eigenvalues efficiently *)
  let eigenvalues, eigenvectors = 
    Tensor_ops.efficient_eigensystem k_matrix model.chunk_size in
  let transformed_eigenvalues = 
    Tensor.map eigenvalues ~f:model.base_model.transform
  in
  
  (* Efficient matrix operations *)
  let k_transformed = 
    if model.use_parallel then
      let v = eigenvectors in
      let d = Tensor.diag transformed_eigenvalues in
      parallel_matmul 
        (parallel_matmul v d model.chunk_size)
        (Tensor.transpose v 0 1) 
        model.chunk_size
    else
      let v = eigenvectors in
      let d = Tensor.diag transformed_eigenvalues in
      Tensor_ops.chunked_matmul 
        (Tensor_ops.chunked_matmul v d model.chunk_size)
        (Tensor.transpose v 0 1) 
        model.chunk_size
  in
  
  (* Solve system *)
  let reg_matrix = Tensor.add k_transformed 
    (Tensor.eye n_labeled |> Tensor.mul model.base_model.params.lambda) in
  let alpha = solve_conjugate_gradient 
    reg_matrix y_labeled 
    model.base_model.params.epsilon 
    model.base_model.params.max_iter in
  
  (x_labeled, alpha)

(* Optimized predict implementation *)
let predict model (x_train, alpha) x_test =
  let compute_kernel_cached x y =
    let key = Kernel_cache.compute_key x y in
    Kernel_cache.get_or_compute model.kernel_cache key 
      model.base_model.kernel x y
  in
  
  let n_test = Tensor.size x_test 0 in
  let predictions = Tensor.zeros [n_test] in
  
  (* Process in chunks *)
  for i = 0 to (n_test + model.chunk_size - 1) / model.chunk_size - 1 do
    let start_idx = i * model.chunk_size in
    let end_idx = min (start_idx + model.chunk_size) n_test in
    let x_chunk = Tensor.narrow x_test 0 start_idx (end_idx - start_idx) in
    
    let k_chunk = compute_kernel_cached x_train x_chunk in
    let pred_chunk = Tensor.matmul k_chunk (Tensor.unsqueeze alpha 1) in
    
    let dest = Tensor.narrow predictions 0 start_idx (end_idx - start_idx) in
    Tensor.copy_ dest (Tensor.squeeze pred_chunk ~dim:[1])
  done;
  
  predictions