open Torch

type t = {
  model: nn;
  betas: Tensor.t;
  alphas: Tensor.t;
  alpha_bars: Tensor.t;
  num_timesteps: int;
  condition_dim: int;
}

let unet_block in_channels out_channels =
  let open Nn in
  sequential [
    conv1d ~in_channels ~out_channels ~kernel_size:3 ~padding:1 ();
    batch_norm1d out_channels;
    relu ();
    conv1d ~in_channels:out_channels ~out_channels ~kernel_size:3 ~padding:1 ();
    batch_norm1d out_channels;
    relu ();
  ]

let create ~num_timesteps ~channels ~condition_dim =
  let betas = Tensor.linspace ~start:0.0001 ~end_:0.02 ~steps:num_timesteps in
  let alphas = Tensor.(ones [num_timesteps] - betas) in
  let alpha_bars = Tensor.cumprod alphas ~dim:0 in
  
  let model = 
    let open Nn in
    sequential [
      unet_block (channels + condition_dim + 1) 64;
      unet_block 64 128;
      unet_block 128 256;
      unet_block 256 128;
      unet_block 128 64;
      conv1d ~in_channels:64 ~out_channels:channels ~kernel_size:3 ~padding:1 ();
    ]
  in
  { model; betas; alphas; alpha_bars; num_timesteps; condition_dim }

let forward t x timestep condition =
  let timestep_emb = Tensor.full [1; 1; Tensor.shape x |> List.nth (-1)] (float_of_int timestep) in
  let condition = Tensor.expand condition ~size:[Tensor.shape x |> List.hd; t.condition_dim; Tensor.shape x |> List.nth (-1)] in
  let input = Tensor.cat [x; timestep_emb; condition] ~dim:1 in
  Nn.forward t.model input

let loss t x condition =
  let noise = Tensor.randn_like x in
  let timestep = Random.int t.num_timesteps in
  let alpha_bar_t = Tensor.slice t.alpha_bars ~dim:0 ~start:timestep ~end_:(timestep + 1) in
  let noisy_x = Tensor.(sqrt alpha_bar_t * x + sqrt (Scalar.f 1. - alpha_bar_t) * noise) in
  let predicted_noise = forward t noisy_x timestep condition in
  Tensor.mse_loss predicted_noise noise ~reduction:Mean

let sample_with_guidance t condition ~num_samples ~seq_length ~guidance_scale =
  let x = Tensor.randn [num_samples; 1; seq_length] in
  let rec loop x t =
    if t < 0 then x
    else
      let z = if t > 0 then Tensor.randn_like x else Tensor.zeros_like x in
      let eps_cond = forward t x t (Some condition) in
      let eps_uncond = forward t x t None in
      let eps = Tensor.((Scalar.f 1. + guidance_scale) * eps_cond - guidance_scale * eps_uncond) in
      let alpha_t = Tensor.slice t.alphas ~dim:0 ~start:t ~end_:(t + 1) in
      let alpha_bar_t = Tensor.slice t.alpha_bars ~dim:0 ~start:t ~end_:(t + 1) in
      let beta_t = Tensor.slice t.betas ~dim:0 ~start:t ~end_:(t + 1) in
      let denominator = Tensor.(sqrt (Scalar.f 1. - alpha_bar_t)) in
      let x_prev = Tensor.((Scalar.f 1. / sqrt alpha_t) * (x - ((Scalar.f 1. - alpha_t) / denominator) * eps) + sqrt beta_t * z) in
      loop x_prev (t - 1)
  in
  loop x (t.num_timesteps - 1)

let train t ~data ~conditions ~learning_rate ~num_epochs =
  let optimizer = Optimizer.adam (Nn.parameters t.model) ~learning_rate in
  for epoch = 1 to num_epochs do
    Tensor.no_grad (fun () ->
      let loss = loss t data conditions in
      Optimizer.backward optimizer loss;
      Optimizer.step optimizer;
      Optimizer.zero_grad optimizer;
      if epoch mod 100 = 0 then
        Stdio.printf "Epoch %d: Loss %.4f\n" epoch (Tensor.float_value loss)
    )
  done