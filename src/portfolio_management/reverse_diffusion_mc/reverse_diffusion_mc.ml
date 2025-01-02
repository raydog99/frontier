open Torch

module Types = struct
  type dimension = {
    input_dim: int;
    batch_size: int;
  }

  type diffusion_params = {
    dt: float;
    total_time: float;
    epsilon: float;
  }

  type score_config = {
    n_samples: int;
    step_size: float;
    n_steps: int;
    threshold: float;
    switch_time: float;
    l2_reg: float;
    grad_clip: float option;
    temperature: float;
  }

  type sample_stats = {
    acceptance_rate: float;
    ess: float;
    grad_norm: float;
    kl_estimate: float option;
  }

  type chain_state = {
    position: Tensor.t;
    log_prob: Tensor.t;
    score: Tensor.t;
    stats: sample_stats;
  }

  type monitor_config = {
    check_interval: int;
    target_r_hat: float;
    min_n_eff: float;
  }
end

(* Numerical stability utilities *)
module Numerical = struct
  let stable_exp x =
    let max_x = Tensor.max x ~dim:[0] ~keepdim:true in
    let shifted_x = Tensor.(x - max_x) in
    Tensor.(exp shifted_x, max_x)

  let logsumexp x =
    let exp_x, max_x = stable_exp x in
    Tensor.(log (sum exp_x) + max_x)

  let stable_softmax x =
    let exp_x, _ = stable_exp x in
    let sum_exp = Tensor.sum exp_x in
    Tensor.(exp_x / sum_exp)

  let clip_by_norm x max_norm =
    let norm = Tensor.norm x ~p:2 in
    let scale = Tensor.(min (ones_like norm) (max_norm / norm)) in
    Tensor.(x * scale)
end

(* Target distribution functor *)
module Target = struct
  module Make(P : sig
    val mean : float array
    val std : float
  end) = struct
    let gaussian_log_density x =
      let dim = Tensor.size x 0 in
      let mean = Tensor.of_float_array P.mean in
      let var = P.std *. P.std in
      let norm_const = -0.5 *. float dim *. Float.log (2. *. Float.pi *. var) in
      let diff = Tensor.(x - mean) in
      Tensor.(
        ((-1. /. (2. *. var)) * (sum (diff * diff) ~dim:[0] ~keepdim:true)) + norm_const
      )

    let log_density = gaussian_log_density
    
    let grad_log_density x =
      let grad_x = Tensor.grad x [log_density x] in
      List.hd grad_x
  end

  (* Gaussian mixture convenience constructor *)
  let make_mixture ~means ~std = 
    let module M = struct
      let components = Array.map (fun mu -> (mu, std)) means
      
      let log_density x =
        let components = Array.map (fun (mu, s) ->
          let mean = Tensor.float_vec [|mu|] in
          let var = s *. s in
          let dim = Tensor.size x 0 in
          let norm_const = -0.5 *. float dim *. Float.log (2. *. Float.pi *. var) in
          let diff = Tensor.(x - mean) in
          Tensor.(
            ((-1. /. (2. *. var)) * (sum (diff * diff) ~dim:[0] ~keepdim:true)) + norm_const
          )
        ) components in
        Numerical.logsumexp (Tensor.stack components)

      let grad_log_density x =
        let grad_x = Tensor.grad x [log_density x] in
        List.hd grad_x
    end in
    (module M : sig
      val log_density : Tensor.t -> Tensor.t
      val grad_log_density : Tensor.t -> Tensor.t
    end)
end

(* Ornstein-Uhlenbeck process *)
module OU_process = struct
  let transition_kernel ~t ~x ~x0 =
    let dim = Tensor.size x 0 in
    let var = 1. -. Float.exp (-2. *. t) in
    let mean = Tensor.(x0 * (exp t)) in
    let diff = Tensor.(x - mean) in
    let norm_factor = Float.pow (2. *. Float.pi *. var) (-0.5 *. float dim) in
    Tensor.(
      exp ((-1. /. (2. *. var)) * (sum (diff * diff) ~dim:[0] ~keepdim:true))
      * norm_factor
    )

  let forward_sde ~t ~x =
    let drift = Tensor.(x) in
    let diffusion = Tensor.ones_like x |> Tensor.mul_scalar (Float.sqrt 2.) in
    drift, diffusion

  let reverse_sde ~t ~x ~score =
    let drift = Tensor.(x + (score * 2.)) in
    let diffusion = Tensor.ones_like x |> Tensor.mul_scalar (Float.sqrt 2.) in
    drift, diffusion

  let scaled_reverse_sde ~t ~x ~score ~temperature =
    let drift = Tensor.(x + (score * 2. * temperature)) in
    let diffusion = 
      Tensor.ones_like x |> Tensor.mul_scalar (Float.sqrt (2. *. temperature)) in
    drift, diffusion
end

(* Score estimation with combined ULA and importance sampling *)
module Score_estimator = struct
  let last_acceptance_rate = ref 1.0

  let estimate_score target ~t ~x ~(config : Types.score_config) =
    let dim = Tensor.size x 0 in

    (* Importance sampling *)
    if t > config.switch_time then
      let mean = Tensor.(x * (exp (-.t))) in
      let var = 1. -. Float.exp (-2. *. t) in
      let samples = Tensor.(
        randn [config.n_samples; dim] * (sqrt var) + mean
      ) in
      
      let log_weights = Tensor.(
        target.log_density samples - 
        (((-1. /. (2. *. var)) * (sum ((samples - mean) * (samples - mean)) ~dim:[0])) +
         (-0.5 *. float dim *. Float.log (2. *. Float.pi *. var)))
      ) in
      
      let weights = Numerical.stable_softmax log_weights in
      
      let diff_term = Tensor.((samples * (exp t) - x) / (1. -. exp (2. *. t))) in
      Tensor.(sum (diff_term * weights))
    
    (* ULA *)
    else
      (* Generate initial samples *)
      let samples = Tensor.(randn [config.n_samples; dim]) in
      
      (* ULA iterations *)
      let rec ula_loop samples step acceptance_sum =
        if step >= config.n_steps then 
          samples, acceptance_sum /. float config.n_steps
        else
          let grad = target.grad_log_density samples in
          let reg_grad = Tensor.(
            grad / (1. +. config.l2_reg *. (norm grad ~p:2))
          ) in
          
          let grad = match config.grad_clip with
          | Some max_norm -> Numerical.clip_by_norm reg_grad max_norm
          | None -> reg_grad
          in
          
          let noise = Tensor.(randn [config.n_samples; dim]) in
          let proposed = Tensor.(
            samples - 
            (grad * config.step_size) +
            ((x - (samples * (exp t))) / (1. -. exp (2. *. t))) * config.step_size +
            (noise * (sqrt (2. *. config.step_size)))
          ) in
          
          (* Compute acceptance rate *)
          let old_log_prob = target.log_density samples in
          let new_log_prob = target.log_density proposed in
          let acceptance = Tensor.(mean (
            min 
              (exp ((new_log_prob - old_log_prob) / config.temperature))
              (ones_like new_log_prob)
          )) |> Tensor.to_float0_exn in
          
          ula_loop proposed (step + 1) (acceptance_sum +. acceptance)
      in
      
      let final_samples, avg_acceptance = 
        ula_loop samples 0 0. in
        
      last_acceptance_rate := avg_acceptance;
      
      (* Compute mean estimate *)
      let weighted_samples = Tensor.(
        (final_samples * (exp t) - x) / (1. -. exp (2. *. t))
      ) in
      Tensor.mean weighted_samples ~dim:[0]

  let get_acceptance_rate () = !last_acceptance_rate
end

module RDMC = struct
  let create_chain_state ~position ~score target =
    let log_prob = target.log_density position in
    {
      Types.position;
      log_prob;
      score;
      stats = {
        acceptance_rate = Score_estimator.get_acceptance_rate ();
        ess = 1.0;
        grad_norm = Tensor.(norm score ~p:2) |> Tensor.to_float0_exn;
        kl_estimate = None;
      }
    }

  let sample 
    target
    ~(config : Types.diffusion_params * Types.score_config)
    ~init_sample =
    
    let diff_params, score_params = config in
    
    let rec outer_loop x_t step =
      if step >= int_of_float (diff_params.total_time /. diff_params.dt)
      then x_t
      else
        let t = diff_params.total_time -. (float step *. diff_params.dt) in
        
        let score = Score_estimator.estimate_score
          target
          ~t
          ~x:x_t
          ~config:score_params in
        
        (* Update with reverse SDE *)
        let drift, diffusion = OU_process.reverse_sde ~t ~x:x_t ~score in
        let noise = Tensor.(randn (shape x_t)) in
        let x_next = Tensor.(
          x_t + 
          (drift * diff_params.dt) +
          (diffusion * noise * (sqrt diff_params.dt))
        ) in
        
        outer_loop x_next (step + 1)
    in
    
    outer_loop init_sample 0

  let sample_gaussian_mixture 
    ~means 
    ~std 
    ~config 
    ~init_sample =
    let target = Target.make_mixture ~means ~std in
    sample target ~config ~init_sample
end