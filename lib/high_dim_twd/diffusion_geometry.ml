open Torch

let build_affinity_matrix points epsilon =
  let n = Tensor.size points 0 in
  let dist_matrix = Tensor.cdist points points in
  Tensor.exp (Tensor.div_scalar 
    (Tensor.neg (Tensor.mul dist_matrix dist_matrix)) epsilon)

let normalize_affinity matrix =
  let row_sums = Tensor.sum matrix ~dim:[1] ~keepdim:true in
  let d_inv = Tensor.pow row_sums (-0.5) in
  let d_inv = Tensor.where (Tensor.isfinite d_inv) 
    d_inv (Tensor.zeros_like d_inv) in
  Tensor.mm (Tensor.mm d_inv matrix) d_inv

let build_diffusion_operator affinity =
  let d = Tensor.sum affinity ~dim:[1] ~keepdim:true in
  let d_inv = Tensor.pow d (-1.0) in
  let d_inv = Tensor.where (Tensor.isfinite d_inv) 
    d_inv (Tensor.zeros_like d_inv) in
  Tensor.mm d_inv affinity

let fast_affinity_matrix points epsilon =
  let n = Tensor.size points 0 in
  let batch_size = min n 1000 in
  let result = Tensor.zeros [n; n] in
  
  for i = 0 to n - 1 step batch_size do
    let batch_end = min (i + batch_size) n in
    let batch = Tensor.narrow points ~dim:0 ~start:i 
      ~length:(batch_end - i) in
    
    let batch_dists = Tensor.cdist batch points in
    let batch_affinity = Tensor.exp (Tensor.div_scalar 
      (Tensor.neg (Tensor.mul batch_dists batch_dists)) epsilon) in
    
    let _ = Tensor.narrow_copy_ result ~dim:0 ~start:i 
      ~length:(batch_end - i) batch_affinity in
    ()
  done;
  result

let stable_normalize_affinity matrix =
  let row_sums = Tensor.sum matrix ~dim:[1] ~keepdim:true in
  let eps = 1e-10 in
  let d_inv = Tensor.pow 
    (Tensor.add row_sums (Tensor.full_like row_sums eps)) (-0.5) in
  Tensor.mm (Tensor.mm d_inv matrix) d_inv