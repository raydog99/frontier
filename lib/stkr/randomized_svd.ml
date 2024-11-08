open Torch

(* Random range finder *)
let range_finder mat n_components n_oversamples =
  let n = Tensor.size mat 0 in
  let l = n_components + n_oversamples in
  (* Standard Gaussian random matrix *)
  let omega = Tensor.randn [n; l] in
  (* Sample the range of mat *)
  Tensor.matmul mat omega

(* QR decomposition with numerical stability *)
let qr_decomp mat =
  let q, _ = Tensor.qr mat ~some:true in
  q

(* Power iteration with QR-based orthogonalization *)
let power_iteration mat y n_iter =
  let rec iter y_mat i =
    if i = 0 then y_mat
    else
      (* Project onto range of mat *)
      let z = Tensor.matmul mat y_mat in
      let q = qr_decomp z in
      (* Project back *)
      let z = Tensor.matmul (Tensor.transpose mat 0 1) q in
      let q = qr_decomp z in
      iter q (i - 1)
  in
  iter y n_iter

(* Main randomized SVD implementation *)
let randomized_svd mat n_components n_oversamples n_iter =
  (* Orthonormal basis for range *)
  let y = range_finder mat n_components n_oversamples in
  let q = power_iteration mat y n_iter |> qr_decomp in
  
  (* Project and perform SVD on smaller matrix *)
  let b = Tensor.matmul (Tensor.transpose q 0 1) mat in
  let u_tilde, s, vt = Tensor.svd b ~some:true in
  let u = Tensor.matmul q u_tilde in
  
  (* Return factorization *)
  u, s, vt