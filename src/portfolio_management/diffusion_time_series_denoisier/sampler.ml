open Torch
open Score_network

let predictor_step sde score_net x t dt =
  let score = Score_network.forward score_net x (Tensor.float_vec [|t|]) in
  let drift = Tensor.(sde.f x t - (float (sde.g t ** 2.)) * score) in
  let diffusion = sde.g t in
  let z = Tensor.randn_like x in
  Tensor.(x + drift * float dt + float diffusion * sqrt (float dt) * z)

let corrector_step sde score_net x t =
  let score = Score_network.forward score_net x (Tensor.float_vec [|t|]) in
  let z = Tensor.randn_like x in
  let eps = 2. *. (sde.g t ** 2.) in
  Tensor.(x + float eps * score + sqrt (float (2. *. eps)) * z)

let sample sde score_net x_T num_steps =
  let dt = 1. /. float num_steps in
  let rec loop x t step =
    if step = 0 then x
    else
      let t_next = max 0. (t -. dt) in
      let x_pred = predictor_step sde score_net x t dt in
      let x_corr = corrector_step sde score_net x_pred t_next in
      loop x_corr t_next (step - 1)
  in
  loop x_T 1. num_steps