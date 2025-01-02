open Torch

let chunked_matmul a b chunk_size =
  let m, n = Tensor.size a 0, Tensor.size b 1 in
  let result = Tensor.zeros [m; n] in
  
  for i = 0 to (m + chunk_size - 1) / chunk_size - 1 do
    let start_i = i * chunk_size in
    let end_i = min (start_i + chunk_size) m in
    let a_chunk = Tensor.narrow a 0 start_i (end_i - start_i) in
    
    for j = 0 to (n + chunk_size - 1) / chunk_size - 1 do
      let start_j = j * chunk_size in
      let end_j = min (start_j + chunk_size) n in
      let b_chunk = Tensor.narrow b 1 start_j (end_j - start_j) in
      
      let result_chunk = Tensor.matmul a_chunk b_chunk in
      let dest = Tensor.narrow result 0 start_i (end_i - start_i) in
      let dest = Tensor.narrow dest 1 start_j (end_j - start_j) in
      Tensor.copy_ dest result_chunk
    done
  done;
  result

let efficient_eigensystem tensor chunk_size =
  let n = Tensor.size tensor 0 in
  if n <= chunk_size then
    Linalg.eigensystem tensor
  else
    let n_components = min chunk_size n in
    Randomized_svd.randomized_svd tensor n_components 10 5