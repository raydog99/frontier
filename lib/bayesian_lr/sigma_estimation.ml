open Torch
open Type

let estimate_sigma x y theta_star =
  let n = float_of_int (size x 0) in
  let residuals = sub y (matmul x theta_star) in
  let sigma_sq_naive = div (dot residuals residuals) n in
  
  (* Scale correction *)
  let sigma_consistent = 
    sqrt (sigma_sq_naive *. (1. -. (log (size x 1 |> float_of_int)) /. n)) in
  sigma_consistent

let estimate_sigma_distributed data theta_star =
  let n = float_of_int data.n in
  let residuals_sq = dot 
    (sub data.xty (matmul data.xtx theta_star))
    theta_star in
  let sigma_sq_naive = div residuals_sq n in
  
  sqrt (sigma_sq_naive *. (1. -. (log (float_of_int data.p)) /. n))

let sample_sigma x y theta_star n =
  let sigma_tilde = estimate_sigma x y theta_star in
  let sigma_hat = sqrt (div (dot (sub y (matmul x theta_star)) 
                             (sub y (matmul x theta_star)))
                      (float_of_int n)) in
  
  (* Generate gamma sample *)
  let shape = float_of_int n /. 2. in
  let scale = sigma_tilde ** 2. *. float_of_int n /. 2. in
  let tau = Random.gamma shape scale in
  
  (* Apply immersion map *)
  let sigma_star = sigma_tilde /. sigma_hat *. (1. /. sqrt tau) in
  sigma_star

let sample_sigma_distributed data theta_star =
  let sigma_tilde = estimate_sigma_distributed data theta_star in
  let residuals_sq = dot 
    (sub data.xty (matmul data.xtx theta_star))
    theta_star in
  let sigma_hat = sqrt (div residuals_sq (float_of_int data.n)) in
  
  let shape = float_of_int data.n /. 2. in
  let scale = sigma_tilde ** 2. *. float_of_int data.n /. 2. in
  let tau = Random.gamma shape scale in
  
  sigma_tilde /. sigma_hat *. (1. /. sqrt tau)