open Torch

type config = {
  noise_level: float;
  num_steps: int;
  num_samples: int;
  corrector_steps: int;
  tv_weight: float;
  fourier_weight: float;
  fourier_threshold: float;
}

let denoise sde score_net x config =
  let denoised_samples = 
    List.init config.num_samples (fun _ ->
      let x_T = sde.sample x config.noise_level (Tensor.randn_like x) in
      let rec loop x t step =
        if step = 0 then x
        else
          let t_next = max 0. (t -. (1. /. float config.num_steps)) in
          let x_pred = Sampler.predictor_step sde score_net x t (1. /. float config.num_steps) in
          let x_corr = 
            List.fold_left (fun acc _ -> 
              Sampler.corrector_step sde score_net acc t_next
            ) x_pred (List.init config.corrector_steps (fun _ -> ()))
          in
          let x_tv = Tensor.(x_corr - float config.tv_weight * Losses.tv_loss x_corr) in
          let x_fourier = Tensor.(x_tv - float config.fourier_weight * 
            Losses.fourier_loss x_tv x config.fourier_threshold)
          in
          loop x_fourier t_next (step - 1)
      in
      loop x_T 1. config.num_steps
    )
  in
  Tensor.mean (Tensor.stack denoised_samples ~dim:0)

let denoise_time_series sde score_net time_series config =
  let denoised_series = 
    Tensor.map_tensors time_series ~f:(fun x -> 
      denoise sde score_net x config
    )
  in
  denoised_series