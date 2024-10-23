open Torch

let frobenius_norm t =
  Tensor.norm t |> Tensor.to_float0_exn

let operator_norm t =
  let s = Tensor.svd_values t in
  Tensor.max s |> Tensor.to_float0_exn

let cholesky_add_jitter t =
  let dim = Tensor.size t |> List.hd in
  let rec try_cholesky jitter =
    try
      let jitter_matrix = Tensor.eye dim |> Tensor.mul_scalar jitter in
      let matrix = Tensor.add t jitter_matrix in
      Tensor.cholesky matrix
    with _ ->
      if jitter > 1e2 then
        failwith "Failed to compute Cholesky decomposition"
      else
        try_cholesky (jitter *. 10.)
  in
  try_cholesky 1e-6

let is_positive_definite t =
  try
    let _ = Tensor.cholesky t in
    true
  with _ -> false

let nearest_positive_definite t =
  let dim = Tensor.size t |> List.hd in
  let sym = Tensor.add t (Tensor.transpose t ~dim0:0 ~dim1:1) |> 
           Tensor.div_scalar 2. in
  let eigenvals, eigenvecs = Tensor.symeig sym ~eigenvectors:true in
  let min_eval = Tensor.min eigenvals |> Tensor.to_float0_exn in
  if min_eval > 0. then sym
  else
    let jitter = Tensor.eye dim |> Tensor.mul_scalar (abs_float min_eval +. 1e-6) in
    Tensor.add sym jitter

let matrix_power t power =
  let eigenvals, eigenvecs = Tensor.symeig t ~eigenvectors:true in
  let powered_vals = Tensor.pow_scalar eigenvals power in
  let diag = Tensor.diag powered_vals in
  let eigenvecs_t = Tensor.transpose eigenvecs ~dim0:0 ~dim1:1 in
  Tensor.mm (Tensor.mm eigenvecs diag) eigenvecs_t