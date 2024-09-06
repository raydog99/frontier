open Torch

let tv_loss x =
  let diff = Tensor.(slice x ~dim:0 ~start:1 ~end:None - slice x ~dim:0 ~start:0 ~end:(-1)) in
  Tensor.sum (Tensor.abs diff)

let fft x =
  Tensor.fft x ~n:None ~dim:(-1) ~norm:None

let ifft x =
  Tensor.ifft x ~n:None ~dim:(-1) ~norm:None

let fourier_loss x_t x f =
  let fft_x_t = fft x_t in
  let fft_x = fft x in
  let filtered_fft_x = Tensor.where (Tensor.abs fft_x > f) fft_x (Tensor.zeros_like fft_x) in
  Tensor.mse_loss fft_x_t filtered_fft_x ~reduction:Torch_core.Reduction.Mean