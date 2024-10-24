open Torch

type device_config = {
  use_gpu: bool;
  device_id: int;
  precision: [`Float | `Double];
}

let get_device config =
  if config.use_gpu && Cuda.is_available () then
    Device.Cuda config.device_id
  else
    Device.Cpu

let to_device t config =
  let device = get_device config in
  let t = Tensor.to_device t device in
  match config.precision with
  | `Float -> Tensor.to_type t Float
  | `Double -> Tensor.to_type t Double

let gpu_batch_mm matrices config =
  let device = get_device config in
  let batch = Tensor.stack matrices ~dim:0 in
  let batch_gpu = to_device batch config in
  let result = Tensor.bmm batch_gpu 
    (Tensor.transpose batch_gpu (-1) (-2)) in
  Tensor.to_device result Device.Cpu

let gpu_covariance samples config =
  let samples_gpu = to_device samples config in
  let mean = Tensor.mean samples_gpu ~dim:[0] ~keepdim:true in
  let centered = Tensor.sub samples_gpu 
    (Tensor.expand_as mean samples_gpu) in
  let cov = Tensor.mm 
    (Tensor.transpose centered 0 1) 
    centered in
  Tensor.to_device cov Device.Cpu

let gpu_svd m config =
  let m_gpu = to_device m config in
  let u, s, v = Tensor.linalg_svd m_gpu ~some:false in
  Tensor.to_device u Device.Cpu,
  Tensor.to_device s Device.Cpu,
  Tensor.to_device v Device.Cpu