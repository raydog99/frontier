open Torch

(* Numerical stability utilities *)
let stable_log x =
  let eps = Tensor.scalar_tensor 1e-10 in
  Tensor.(log (maximum x eps))

let logsumexp x =
  let max_x = Tensor.max x in
  let shifted = Tensor.(x - max_x) in
  Tensor.(max_x + log (sum (exp shifted)))

let log_normalize log_weights =
  let lse = logsumexp log_weights in  
  Tensor.(log_weights - lse)

let normalize tensor =
  let sum = Tensor.sum tensor in
  let eps = Tensor.scalar_tensor 1e-10 in
  Tensor.(tensor / maximum sum eps)

(* Matrix utilities *)
let is_pos_def mat =
  try
    let _ = Tensor.cholesky mat in
    true
  with _ -> false

let ensure_pos_def mat =
  let n = (Tensor.size mat).(0) in
  let eps = Tensor.scalar_tensor 1e-6 in
  let eye = Tensor.(eps * eye n) in
  if is_pos_def mat then mat
  else Tensor.(mat + eye)

(* Distribution utilities *)
let kl_divergence p q =
  let log_p = stable_log p in
  let log_q = stable_log q in
  Tensor.(sum (p * (log_p - log_q)))

let mvnormal mean covar =
  let l = Tensor.cholesky covar in
  let z = Tensor.randn (Tensor.size mean) in
  Tensor.(mean + mm l z)

(* Resampling utilities *)
let systematic_resample weights n =
  let cumsum = Tensor.(cumsum weights ~dim:0) in
  let step = Tensor.scalar_tensor (1. /. float_of_int n) in
  let u = Tensor.(rand [1] + step) in
  let indices = Tensor.empty [n] ~kind:(T Int64) in
  
  for i = 0 to n - 1 do
    let threshold = Tensor.(u + (Scalar.f (float_of_int i) * step)) in
    let idx = Tensor.(sum (cumsum < threshold)) in
    Tensor.set indices [i] idx
  done;
  indices