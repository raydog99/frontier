open Torch
open Types

type regularization = 
  | Tikhonov of float
  | Truncated of float
  | Adaptive of float

let estimate_condition_number m =
  let s = Tensor.linalg_svdvals m in
  let max_s = Tensor.max s |> fst |> Tensor.item in
  let min_s = Tensor.min s |> fst |> Tensor.item in
  max_s /. min_s

let regularize m reg device_config =
  let m = Gpu_compute.to_device m device_config in
  match reg with
  | Tikhonov lambda ->
      let d = Tensor.size m 0 in
      Tensor.add m (Tensor.mul_scalar (Tensor.eye d) lambda)
  | Truncated threshold ->
      let u, s, vh = Gpu_compute.gpu_svd m device_config in
      let s = Tensor.maximum s (Tensor.full_like s threshold) in
      Tensor.mm u (Tensor.mm (Tensor.diag s) vh)
  | Adaptive target_cond ->
      let s = Tensor.linalg_svdvals m in
      let max_s = Tensor.max s |> fst |> Tensor.item in
      let min_s = Tensor.min s |> fst |> Tensor.item in
      let current_cond = max_s /. min_s in
      if current_cond <= target_cond then m
      else
        let lambda = max_s *. (1. /. target_cond -. min_s /. max_s) in
        regularize m (Tikhonov lambda) device_config

let robust_covariance_estimation samples reg device_config =
  let cov = Gpu_compute.gpu_covariance samples device_config in
  let condition = estimate_condition_number cov in
  
  if condition > 1e6 then
    let reg_cov = regularize cov reg device_config in
    {
      mean = Tensor.mean samples ~dim:[0] ~keepdim:false;
      covariance = reg_cov;
    }
  else
    {
      mean = Tensor.mean samples ~dim:[0] ~keepdim:false;
      covariance = cov;
    }