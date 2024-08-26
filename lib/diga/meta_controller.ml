open Torch
open Diffusion_model
open Control_encoder

  type t = {
    diffusion_model: Diffusion_model.t;
    control_encoder: Control_encoder.t;
    num_samples: int;
    seq_length: int;
    condition_dim: int;
  }

  let create ~num_timesteps ~channels ~num_samples ~seq_length ~condition_dim ~encoder_type =
    let control_encoder = match encoder_type with
      | `Discrete num_classes -> Control_encoder.create_discrete ~num_classes ~condition_dim
      | `Continuous input_dim -> Control_encoder.create_continuous ~input_dim ~condition_dim
    in
    {
      diffusion_model = Diffusion_model.create ~num_timesteps ~channels ~condition_dim;
      control_encoder;
      num_samples;
      seq_length;
      condition_dim;
    }

  let train t ~data ~conditions ~learning_rate ~num_epochs =
    let encoded_conditions = Control_encoder.encode t.control_encoder conditions in
    Diffusion_model.train t.diffusion_model ~data ~conditions:encoded_conditions ~learning_rate ~num_epochs

  let generate_market_states t control_target ~guidance_scale =
    let encoded_condition = Control_encoder.encode t.control_encoder control_target in
    Diffusion_model.sample_with_guidance t.diffusion_model encoded_condition ~num_samples:t.num_samples ~seq_length:t.seq_length ~guidance_scale