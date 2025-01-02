open Torch

let qr_decomposition matrix =
  let q, r = Tensor.qr matrix ~some:false in
  q, r

let compute_residuals matrix beta response =
  let pred = Tensor.mm matrix beta in
  Tensor.sub response pred

let sample_matrix matrix probs sample_size =
  let indices = Tensor.multinomial probs ~num_samples:sample_size ~replacement:true in
  Tensor.index_select matrix ~dim:0 ~index:indices

let condition_number matrix =
  let singular_vals = Tensor.svd matrix ~some:true in
  let max_sv = Tensor.max singular_vals in
  let min_sv = Tensor.min singular_vals in
  Tensor.div max_sv min_sv

let batch_matmul a b batch_size =
  let m, n = Tensor.shape2_exn a in
  let result = Tensor.zeros [m; Tensor.shape2_exn b] in
  
  let num_batches = (m + batch_size - 1) / batch_size in
  for i = 0 to num_batches - 1 do
    let start_idx = i * batch_size in
    let end_idx = min (start_idx + batch_size) m in
    let batch_a = Tensor.narrow a ~dim:0 ~start:start_idx ~length:(end_idx - start_idx) in
    let batch_result = Tensor.mm batch_a b in
    Tensor.copy_ 
      (Tensor.narrow result ~dim:0 ~start:start_idx ~length:(end_idx - start_idx))
      batch_result
  done;
  result

let solve_regularized_ls a b lambda =
  let m, n = Tensor.shape2_exn a in
  let at = Tensor.transpose a ~dim0:0 ~dim1:1 in
  let reg = Tensor.mul_scalar (Tensor.eye n) lambda in
  let lhs = Tensor.add (Tensor.mm at a) reg in
  let rhs = Tensor.mv at b in
  let q, r = qr_decomposition lhs in
  Tensor.triangular_solve rhs r ~upper:true ~transpose:false ~unitriangular:false

let parallel_matmul a b num_domains =
  let m, n = Tensor.shape2_exn a in
  let result = Tensor.zeros [m; Tensor.shape2_exn b] in
  
  let batch_size = (m + num_domains - 1) / num_domains in
  let domains = Array.init num_domains (fun i ->
    let start_idx = i * batch_size in
    let end_idx = min (start_idx + batch_size) m in
    Domain.spawn (fun () ->
      let batch_a = Tensor.narrow a ~dim:0 ~start:start_idx ~length:(end_idx - start_idx) in
      let batch_result = Tensor.mm batch_a b in
      (start_idx, batch_result)
    )
  ) in
  
  Array.iter (fun domain ->
    let start_idx, batch_result = Domain.join domain in
    Tensor.copy_ 
      (Tensor.narrow result ~dim:0 ~start:start_idx ~length:(Tensor.shape1_exn batch_result))
      batch_result
  ) domains;
  result

let memory_efficient_qr matrix block_size =
  let m, n = Tensor.shape2_exn matrix in
  let q = Tensor.zeros [m; n] in
  let r = Tensor.zeros [n; n] in
  
  let num_blocks = (m + block_size - 1) / block_size in
  for i = 0 to num_blocks - 1 do
    let start_idx = i * block_size in
    let end_idx = min (start_idx + block_size) m in
    let block = Tensor.narrow matrix ~dim:0 ~start:start_idx ~length:(end_idx - start_idx) in
    let q_block, r_block = qr_decomposition block in
    
    Tensor.copy_ 
      (Tensor.narrow q ~dim:0 ~start:start_idx ~length:(end_idx - start_idx))
      q_block;
      
    if i = 0 then
      Tensor.copy_ r r_block
    else
      Tensor.add_ r r_block
  done;
  q, r