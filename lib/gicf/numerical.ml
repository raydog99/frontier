open Torch

let safe_cholesky sigma =
  let n = (Tensor.shape sigma).(0) in
  let jitter = ref 1e-12 in
  
  let rec attempt_cholesky mat tries =
    if tries = 0 then Error "Failed to compute Cholesky decomposition"
    else
      try
        Ok (Tensor.cholesky mat ~upper:false)
      with _ ->
        let adjusted = Tensor.add mat 
          (Tensor.eye n |> Tensor.mul_scalar !jitter) in
        jitter := !jitter *. 10.0;
        attempt_cholesky adjusted (tries - 1)
  in
  attempt_cholesky sigma 10

let safe_inverse sigma =
  match safe_cholesky sigma with
  | Error msg -> Error msg
  | Ok chol ->
      try
        Ok (Tensor.cholesky_inverse sigma)
      with _ -> Error "Failed to compute inverse"

let monitor_conditioning sigma threshold =
  let eigenvals = Tensor.symeig sigma ~eigenvectors:false in
  let max_eig = Tensor.max eigenvals |> Tensor.item in
  let min_eig = Tensor.min eigenvals |> Tensor.item in
  let cond = max_eig /. min_eig in
  
  if cond > threshold then
    Error (Printf.sprintf "Poor conditioning detected: %f" cond)
  else
    Ok sigma

let scale_computations sigma =
  let scaling_factor = Tensor.diagonal sigma ~dim1:0 ~dim2:1 
                      |> Tensor.mean |> Tensor.item in
  let scaled_sigma = Tensor.div_scalar sigma scaling_factor in
  scaled_sigma, scaling_factor

let safe_logdet sigma =
  match safe_cholesky sigma with
  | Error msg -> Error msg
  | Ok chol ->
      let diag = Tensor.diagonal chol ~dim1:0 ~dim2:1 in
      let logdet = Tensor.log diag |> Tensor.sum |> Tensor.mul_scalar 2.0 in
      Ok (Tensor.item logdet)

let stabilize_matrix m epsilon =
  let eigenvals = Tensor.symeig m ~eigenvectors:false in
  let min_eig = Tensor.min eigenvals |> Tensor.item in
  if min_eig < epsilon then
    Tensor.add m (Tensor.eye (Tensor.shape m).(0) |> 
                 Tensor.mul_scalar (epsilon -. min_eig))
  else
    m