open Torch

let add_gaussian_noise tensor noise_level =
  let noise = Tensor.randn (Tensor.shape tensor) in
  Tensor.(tensor + noise_level * noise)

let random_scale tensor scale_range =
  let scale = Tensor.uniform_float ~a:scale_range.0 ~b:scale_range.1 [1] in
  Tensor.(tensor * scale)

let time_warp tensor max_warp =
  let length = Tensor.shape tensor |> List.hd in
  let warp = Tensor.randint ~high:max_warp ~size:[1] |> Tensor.to_int0_exn in
  let start = Random.int (length - warp) in
  let warped = Tensor.narrow tensor ~dim:0 ~start ~length:warp in
  let stretched = Tensor.interpolate warped [length] ~mode:"linear" ~align_corners:false in
  Tensor.cat [Tensor.narrow tensor ~dim:0 ~start:0 ~length:start;
              stretched;
              Tensor.narrow tensor ~dim:0 ~start:(start + warp) ~length:(length - start - warp)] ~dim:0

let augment_tensor tensor =
  tensor
  |> add_gaussian_noise 0.01
  |> random_scale (0.95, 1.05)
  |> time_warp 5