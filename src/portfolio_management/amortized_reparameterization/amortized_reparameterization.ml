open Torch

type stability_config = {
  eps: float;
  min_clamp: float;
  max_clamp: float;
  condition_threshold: float;
}

type sde_params = {
  drift: Tensor.t -> float -> Tensor.t;  (* f_theta(z,t) *)
  dispersion: float -> Tensor.t;         (* L_theta(t) *)
  initial_dist: unit -> Tensor.t * Tensor.t (* mean, covariance *)
}

type variational_params = {
  encoder_mean: Tensor.t list -> float -> Tensor.t;    (* mu_phi(t) *)
  encoder_cov: Tensor.t list -> float -> Tensor.t;     (* Sigma_phi(t) *)
  time_encoder: float -> Tensor.t                      (* time embedding *)
}

let default_stability_config = {
  eps = 1e-6;
  min_clamp = -1e6;
  max_clamp = 1e6;
  condition_threshold = 1e4;
}

let ensure_2d tensor =
  match Tensor.size tensor with
  | [|_; _|] -> tensor
  | [|n|] -> Tensor.reshape tensor ~shape:[n; 1]
  | _ -> failwith "Invalid tensor dimensions"

let batch_matmul a b =
  let a' = ensure_2d a in
  let b' = ensure_2d b in
  Tensor.matmul a' b'

let safe_inverse ?(config=default_stability_config) mat =
  try 
    Tensor.inverse mat
  with _ ->
    let n = (Tensor.size mat).(0) in
    let eps = Tensor.(eye n * float_of_int config.eps) in
    Tensor.(inverse (mat + eps))

let safe_cholesky ?(config=default_stability_config) mat =
  try
    Tensor.cholesky mat
  with _ ->
    let n = (Tensor.size mat).(0) in
    let eps = Tensor.(eye n * float_of_int config.eps) in
    Tensor.cholesky Tensor.(mat + eps)

let safe_log x eps =
  Tensor.(log (x + float_of_int eps))

let safe_div num denom eps =
  Tensor.(num / (denom + float_of_int eps))

let safe_sqrt x eps =
  Tensor.(sqrt (x + float_of_int eps))

let stable_softmax x dim =
  let max_x = Tensor.max x ~dim |> fst in
  let shifted = Tensor.(x - max_x) in
  let exp_x = Tensor.exp shifted in
  let sum_exp = Tensor.sum exp_x ~dim in
  Tensor.(exp_x / sum_exp)

(* Kronecker operations *)
let kron a b =
  let m1, n1 = Tensor.size2 a in
  let m2, n2 = Tensor.size2 b in
  let result = Tensor.zeros [m1 * m2; n1 * n2] in
  
  for i = 0 to m1 - 1 do
    for j = 0 to n1 - 1 do
      let a_ij = Tensor.get a [|i; j|] in
      let sub_matrix = Tensor.(b * float_of_int a_ij) in
      let start_i = i * m2 in
      let start_j = j * n2 in
      Tensor.index_put_ result 
        ~indices:[Tensor.range start_i (start_i + m2);
                 Tensor.range start_j (start_j + n2)]
        sub_matrix
    done
  done;
  result

let kron_mv a b x =
  let m1, n1 = Tensor.size2 a in
  let m2, n2 = Tensor.size2 b in
  let x_mat = Tensor.reshape x ~shape:[n2; n1] in
  let temp = Tensor.matmul b x_mat in
  let temp = Tensor.transpose temp ~dim0:0 ~dim1:1 in
  let result = Tensor.matmul a temp in
  Tensor.reshape result ~shape:[m1 * m2; 1]

let kron_sum a b =
  let m1, n1 = Tensor.size2 a in
  let m2, n2 = Tensor.size2 b in
  let eye1 = Tensor.eye m1 in
  let eye2 = Tensor.eye m2 in
  let sum1 = kron a eye2 in
  let sum2 = kron eye1 b in
  Tensor.(sum1 + sum2)

module SDECore = struct
  type sde_spec = {
    dim: int;
    latent_dim: int;
    hidden_dim: int;
    time_embedding_dim: int;
  }

  type t = {
    spec: sde_spec;
    drift_net: Tensor.t -> Tensor.t -> Tensor.t;
    diffusion_net: Tensor.t -> Tensor.t;
    time_embed_net: float -> Tensor.t;
  }

  let make_time_embedding dim hidden_dim =
    let module N = Neural in
    let net = N.Sequential.make [
      N.Linear.make 1 hidden_dim;
      N.Tanh.make ();
      N.Linear.make hidden_dim dim;
    ] in
    N.Sequential.forward net

  let make_drift_net latent_dim hidden_dim time_dim =
    let module N = Neural in
    let net = N.Sequential.make [
      N.Linear.make (latent_dim + time_dim) hidden_dim;
      N.Tanh.make ();
      N.Linear.make hidden_dim hidden_dim;
      N.Tanh.make ();
      N.Linear.make hidden_dim latent_dim;
    ] in
    fun z t -> 
      let combined = Tensor.cat [z; t] ~dim:1 in
      N.Sequential.forward net combined

  let make_diffusion_net latent_dim hidden_dim =
    let module N = Neural in
    let net = N.Sequential.make [
      N.Linear.make 1 hidden_dim;
      N.Tanh.make ();
      N.Linear.make hidden_dim (latent_dim * latent_dim);
      N.Softplus.make ();
    ] in
    fun t ->
      let output = N.Sequential.forward net t in
      Tensor.reshape output ~shape:[latent_dim; latent_dim]

  let make spec =
    let time_embed_net = make_time_embedding 
      spec.time_embedding_dim spec.hidden_dim in
    let drift_net = make_drift_net 
      spec.latent_dim spec.hidden_dim spec.time_embedding_dim in
    let diffusion_net = make_diffusion_net 
      spec.latent_dim spec.hidden_dim in
    { spec; drift_net; diffusion_net; time_embed_net }

  let drift t z time =
    let batch_size = (Tensor.size z).(0) in
    let time_tensor = Tensor.of_float1 [|time|] in
    let time_embed = t.time_embed_net time_tensor in
    let time_embed = Tensor.repeat time_embed ~repeats:[batch_size; 1] in
    t.drift_net z time_embed

  let diffusion t time =
    let time_tensor = Tensor.of_float1 [|time|] in
    t.diffusion_net time_tensor

  let sample_initial t batch_size =
    let mean = Tensor.zeros [batch_size; t.spec.latent_dim] in
    let std = Tensor.ones [batch_size; t.spec.latent_dim] in
    let eps = Tensor.randn_like mean in
    Tensor.(mean + (eps * std))
end

(* Markov Gaussian Process *)
module MGPCore = struct
  type t = {
    dim: int;
    hidden_dim: int;
    a_net: Tensor.t -> Tensor.t;  (* A(t) *)
    b_net: Tensor.t -> Tensor.t;  (* b(t) *)
  }

  let make dim hidden_dim =
    let module N = Neural in
    
    (* Network for A(t) *)
    let a_net = N.Sequential.make [
      N.Linear.make 1 hidden_dim;
      N.Tanh.make ();
      N.Linear.make hidden_dim (dim * dim);
    ] in
    
    (* Network for b(t) *)
    let b_net = N.Sequential.make [
      N.Linear.make 1 hidden_dim;
      N.Tanh.make ();
      N.Linear.make hidden_dim dim;
    ] in

    {
      dim;
      hidden_dim;
      a_net = N.Sequential.forward a_net;
      b_net = N.Sequential.forward b_net;
    }

  let compute_drift_params t time batch_size =
    let time_tensor = Tensor.of_float1 [|time|] in
    let a_flat = t.a_net time_tensor in
    let a = Tensor.reshape a_flat ~shape:[t.dim; t.dim] in
    let b = t.b_net time_tensor in
    let b = Tensor.reshape b ~shape:[1; t.dim] in
    let b = Tensor.repeat b ~repeats:[batch_size; 1] in
    a, b

  let compute_derivatives t mean cov time =
    let batch_size = (Tensor.size mean).(0) in
    let a, b = compute_drift_params t time batch_size in
    
    (* Mean derivative *)
    let mean_dot = Tensor.(
      matmul mean a + b
    ) in
    
    (* Covariance derivative *)
    let cov_dot = Tensor.(
      matmul (matmul a cov) (transpose a ~dim0:0 ~dim1:1) +
      matmul cov (transpose a ~dim0:0 ~dim1:1)
    ) in
    
    mean_dot, cov_dot

  let initialize t batch_size =
    let mean = Tensor.zeros [batch_size; t.dim] in
    let cov = Tensor.eye t.dim in
    let cov = Tensor.repeat cov ~repeats:[batch_size; 1; 1] in
    mean, cov

  (* Ensure stability in covariance updates *)
  let stable_covariance_update t cov drift diffusion config =
    let stabilized_cov = 
      Tensor.(cov + (eye (size1 cov) * float_of_int config.eps)) in
    let drift_term = Tensor.(
      matmul (matmul drift stabilized_cov) 
             (transpose drift ~dim0:0 ~dim1:1)
    ) in
    let diff_term = Tensor.(
      matmul diffusion (transpose diffusion ~dim0:0 ~dim1:1)
    ) in
    Tensor.(drift_term + diff_term)
end

(* Amortization strategy *)
module Amortization = struct
  type t = {
    partition_size: int;
    overlap: int;
    latent_dim: int;
    hidden_dim: int;
    kernel_net: Tensor.t -> Tensor.t -> Tensor.t;
    encoder_net: Tensor.t -> Tensor.t;
  }

  let make ~partition_size ~overlap ~latent_dim ~hidden_dim =
    let module N = Neural in
    
    (* Deep kernel network for interpolation *)
    let kernel = N.Sequential.make [
      N.Linear.make (2 * hidden_dim) hidden_dim;
      N.Tanh.make ();
      N.Linear.make hidden_dim 1;
      N.Softplus.make ();
    ] in

    (* Encoder network for single observations *)
    let encoder = N.Sequential.make [
      N.Linear.make latent_dim hidden_dim;
      N.ReLU.make ();
      N.Linear.make hidden_dim hidden_dim;
    ] in

    {
      partition_size;
      overlap;
      latent_dim;
      hidden_dim;
      kernel_net = N.Sequential.forward kernel;
      encoder_net = N.Sequential.forward encoder;
    }

  (* Partition handling *)
  let create_partitions t data times =
    let n = List.length data in
    let rec build_partitions acc start =
      if start >= n then List.rev acc
      else
        let end_idx = min (start + t.partition_size) n in
        let overlap_start = max 0 (start - t.overlap) in
        let partition_data = List.sub data overlap_start (end_idx - overlap_start) in
        let partition_times = List.sub times overlap_start (end_idx - overlap_start) in
        build_partitions ((partition_data, partition_times) :: acc) 
          (start + t.partition_size - t.overlap)
    in
    build_partitions [] 0

  (* Deep kernel interpolation *)
  let interpolate t encoded_states times query_time =
    let batch_size = List.length encoded_states in
    let query_times = Tensor.ones [batch_size; 1] |> 
                     Tensor.(mul_scalar (Float.of_float query_time)) in
    
    (* Compute kernel weights *)
    let weights = List.map2 (fun state time ->
      let time_tensor = Tensor.of_float1 [|time|] in
      t.kernel_net state (t.encoder_net time_tensor)
    ) encoded_states times in
    
    let weights = Tensor.stack weights ~dim:0 in
    let states = Tensor.stack encoded_states ~dim:0 in
    
    (* Normalize weights with stable softmax *)
    let weights = stable_softmax weights 0 in
    
    (* Compute weighted sum *)
    Tensor.(sum (weights * states) ~dim:[0])
end

(* Temporal context handling *)
module TemporalContext = struct
  type t = {
    window_size: int;
    stride: int;
    kernel_size: int;
    hidden_dim: int;
    attention_heads: int;
  }

  let make ~window_size ~stride ~kernel_size ~hidden_dim ~attention_heads =
    { window_size; stride; kernel_size; hidden_dim; attention_heads }

  (* Multi-head attention *)
  let attention_module input_dim hidden_dim num_heads =
    let module N = Neural in
    let make_head () = N.Sequential.make [
      N.Linear.make input_dim hidden_dim;
      N.ReLU.make ();
      N.Linear.make hidden_dim hidden_dim;
    ] in
    let heads = List.init num_heads (fun _ -> make_head ()) in
    let combine = N.Linear.make (hidden_dim * num_heads) hidden_dim in
    fun x ->
      let head_outputs = List.map (fun head -> 
        N.Sequential.forward head x) heads in
      let combined = Tensor.cat head_outputs ~dim:1 in
      N.Linear.forward combine combined

  (* Temporal convolution *)
  let temporal_conv_module input_dim kernel_size =
    let module N = Neural in
    N.Sequential.make [
      N.Conv1d.make input_dim input_dim kernel_size 
        ~padding:(kernel_size / 2);
      N.ReLU.make ();
      N.Conv1d.make input_dim input_dim 1;
    ]

  (* Extract temporal neighbors with padding *)
  let extract_neighbors t sequence idx =
    let n = List.length sequence in
    let half_window = t.window_size / 2 in
    let start_idx = max 0 (idx - half_window) in
    let end_idx = min n (idx + half_window + 1) in
    
    let pad_left = max 0 (half_window - idx) in
    let pad_right = max 0 (idx + half_window + 1 - n) in
    
    let padding = 
      if pad_left > 0 then List.init pad_left (fun _ -> List.hd sequence)
      else [] in
    let post_padding = 
      if pad_right > 0 then 
        List.init pad_right (fun _ -> List.hd (List.rev sequence))
      else [] in
    
    padding @ 
    List.sub sequence start_idx (end_idx - start_idx) @
    post_padding

  (* Process sequence with temporal attention *)
  let process_sequence t sequence =
    let attention = attention_module t.hidden_dim t.hidden_dim t.attention_heads in
    let conv = temporal_conv_module t.hidden_dim t.kernel_size in
    
    List.mapi (fun i _ ->
      let neighbors = extract_neighbors t sequence i in
      let neighbor_tensor = Tensor.stack neighbors ~dim:0 in
      
      (* Apply attention and temporal convolution *)
      let attended = attention neighbor_tensor in
      let conv_input = Tensor.unsqueeze attended ~dim:0 in
      let conv_output = N.Sequential.forward conv conv_input in
      Tensor.squeeze conv_output ~dim:0
    ) sequence
end

(* Gradient estimation *)
module GradientEstimation = struct

  type t = {
    outer_samples: int;  (* R *)
    inner_samples: int;  (* S *)
    stratification: bool;
    batch_size: int;
  }

  let make ~outer_samples ~inner_samples ~stratification ~batch_size =
    { outer_samples; inner_samples; stratification; batch_size }

  (* Generate time samples with optional stratification *)
  let generate_time_samples t time_window =
    let t_start, t_end = time_window in
    let delta = (t_end -. t_start) /. Float.of_int t.inner_samples in
    
    if t.stratification then
      (* Stratified sampling *)
      List.init t.inner_samples (fun i ->
        let bin_start = t_start +. delta *. Float.of_int i in
        let u = Random.float 1.0 in
        bin_start +. delta *. u
      )
    else
      (* Uniform sampling *)
      List.init t.inner_samples (fun _ ->
        t_start +. Random.float (t_end -. t_start)
      )

  (* Compute residual term *)
  let compute_residual sde encoder z t =
    let mean, cov = encoder z t in
    let drift = sde.drift z t in
    let diff = Tensor.(drift - mean) in
    diff, cov

  (* Nested Monte Carlo estimation *)
  let estimate_integral t sde encoder time_window =
    let time_samples = generate_time_samples t time_window in
    
    (* Outer loop - sample from variational distribution *)
    let outer_estimates = List.init t.outer_samples (fun _ ->
      let eps = Tensor.randn [t.batch_size; sde.latent_dim] in
      
      (* Inner loop - sample time points *)
      let inner_estimates = List.map (fun time ->
        let z = encoder eps time in
        let residual, cov = compute_residual sde encoder z time in
        Tensor.(residual * residual / cov)
      ) time_samples in
      
      List.fold_left Tensor.(+) (Tensor.zeros [t.batch_size; sde.latent_dim]) 
        inner_estimates
    ) in
    
    (* Average estimates *)
    let sum = List.fold_left Tensor.(+) 
      (Tensor.zeros [t.batch_size; sde.latent_dim]) outer_estimates in
    Tensor.(sum / float_of_int (t.outer_samples * t.inner_samples))

  (* Estimate gradients *)
  let estimate_gradients t sde encoder data time_window =
    let t_start, t_end = time_window in
    
    (* First term - reconstruction *)
    let recon_grads = List.mapi (fun i x ->
      let time = t_start +. (t_end -. t_start) *. 
                Float.of_int i /. Float.of_int (List.length data) in
      let eps = Tensor.randn [1; sde.latent_dim] in
      let z = encoder eps time in
      let residual, _ = compute_residual sde encoder z time in
      residual
    ) data in
    
    (* Second term - trajectory matching using nested Monte Carlo *)
    let traj_grad = estimate_integral t sde encoder time_window in
    
    (* Combine terms *)
    let scale = (t_end -. t_start) /. 
                Float.of_int (2 * List.length data) in
    
    List.map (fun recon -> 
      Tensor.(recon - (traj_grad * float_of_int scale))) recon_grads
end

(* Monte Carlo *)
module MonteCarlo = struct
  type t = {
    parallel: bool;
    batch_size: int;
    num_samples: int;
  }

  let make ~parallel ~batch_size ~num_samples =
    { parallel; batch_size; num_samples }

  (* Sample from normal distribution *)
  let sample_normal mean std =
    let eps = Tensor.randn_like mean in
    Tensor.(mean + (eps * std))

  (* Parallel sampling when enabled *)
  let parallel_sample t f samples =
    if t.parallel then
      let chunks = List.split_n t.batch_size samples in
      List.concat_map (fun chunk ->
        List.map f chunk |> 
        List.map (fun tensor -> Tensor.detach tensor)
      ) chunks
    else
      List.map f samples

  (* Monte Carlo integration *)
  let integrate t f domain num_samples =
    let samples = List.init num_samples (fun _ ->
      let u = Random.float 1.0 in
      let start, end_ = domain in
      start +. u *. (end_ -. start)
    ) in
    
    let evaluations = parallel_sample t f samples in
    let sum = List.fold_left Tensor.(+) 
      (Tensor.zeros [t.batch_size]) evaluations in
    Tensor.(sum / float_of_int num_samples)

  (* Importance sampling *)
  let importance_sampling t f proposal_dist domain num_samples =
    let samples = List.init num_samples (fun _ ->
      proposal_dist ()
    ) in
    
    let weights = List.map (fun x ->
      let px = 1.0 /. (snd domain -. fst domain) in
      let qx = proposal_dist x in
      px /. qx
    ) samples in
    
    let evaluations = parallel_sample t f samples in
    let weighted_sum = List.fold_left2 (fun acc eval w ->
      Tensor.(acc + (eval * float_of_int w))
    ) (Tensor.zeros [t.batch_size]) evaluations weights in
    Tensor.(weighted_sum / float_of_int num_samples)
end

(* Extended optimization *)
module ExtendedOptimization = struct
  type scheduler_type =
    | CosineAnnealing of {min_lr: float; max_lr: float; cycle_steps: int}
    | LinearWarmup of {warmup_steps: int; peak_lr: float}
    | CyclicalLR of {base_lr: float; max_lr: float; step_size: int}

  type t = {
    scheduler: scheduler_type;
    clip_grad_norm: float option;
    weight_decay: float;
    ema_decay: float option;
  }

  let make ~scheduler ~clip_grad_norm ~weight_decay ~ema_decay =
    { scheduler; clip_grad_norm; weight_decay; ema_decay }

  (* Learning rate scheduling *)
  let get_lr t step =
    match t.scheduler with
    | CosineAnnealing {min_lr; max_lr; cycle_steps} ->
        let cycle = float_of_int step /. float_of_int cycle_steps in
        let cosine = (1.0 +. cos (Float.pi *. cycle)) /. 2.0 in
        min_lr +. (max_lr -. min_lr) *. cosine
    | LinearWarmup {warmup_steps; peak_lr} ->
        if step < warmup_steps then
          peak_lr *. float_of_int step /. float_of_int warmup_steps
        else peak_lr
    | CyclicalLR {base_lr; max_lr; step_size} ->
        let cycle = float_of_int (step / (2 * step_size)) in
        let x = float_of_int (step - (int_of_float cycle *. 2.0 *. 
                float_of_int step_size)) /. float_of_int step_size in
        if x <= 1.0 then
          base_lr +. (max_lr -. base_lr) *. x
        else
          max_lr -. (max_lr -. base_lr) *. (x -. 1.0)

  let update_ema ema_params current_params decay =
    List.iter2 (fun ema current ->
      let updated = Tensor.(ema * float_of_int decay + 
                          current * float_of_int (1.0 -. decay)) in
      Tensor.copy_ ema updated
    ) ema_params current_params

  let optimization_step t vs step =
    (* Apply gradient clipping *)
    (match t.clip_grad_norm with
    | Some max_norm -> Optimizer.clip_grad_norm_ vs ~max_norm
    | None -> ());

    (* Update learning rate *)
    let lr = get_lr t step in
    Optimizer.set_learning_rate vs ~learning_rate:lr;

    (* Apply weight decay *)
    if t.weight_decay > 0.0 then
      List.iter (fun var ->
        let tensor = Var_store.tensor var in
        Tensor.(copy_ tensor (tensor * float_of_int (1.0 -. t.weight_decay)))
      ) (Var_store.all_vars vs);

    (* Update EMA if specified *)
    (match t.ema_decay with
    | Some decay ->
        let current_params = List.map Var_store.tensor (Var_store.all_vars vs) in
        let ema_params = List.map Tensor.copy current_params in
        update_ema ema_params current_params decay
    | None -> ());

    Optimizer.step vs;
    Optimizer.zero_grad vs
end

(* Trainer *)
module Trainer = struct
  type config = {
    epochs: int;
    batch_size: int;
    partition_size: int;
    optimizer: ExtendedOptimization.t;
    grad_estimator: GradientEstimation.t;
    checkpoint_interval: int;
    early_stopping_patience: int;
  }

  type training_state = {
    mutable step: int;
    mutable best_loss: float;
    mutable patience_counter: int;
    mutable should_stop: bool;
  }

  let make_training_state () = {
    step = 0;
    best_loss = Float.infinity;
    patience_counter = 0;
    should_stop = false;
  }

  (* Process single batch *)
  let process_batch model grad_estimator batch =
    let data, times, indices = batch in
    
    (* Create partitions *)
    let partitioned = Amortization.create_partitions 
      model.amortization data times in
    
    (* Process each partition *)
    List.concat_map (fun (part_data, part_times) ->
      let window = (List.hd part_times, List.hd (List.rev part_times)) in
      GradientEstimation.estimate_gradients 
        grad_estimator model.sde model.encoder part_data window
    ) partitioned

  (* Training loop with early stopping *)
  let train config model trajectories =
    let state = make_training_state () in
    
    let rec epoch_loop epoch =
      if epoch >= config.epochs || state.should_stop then ()
      else begin
        let total_loss = ref 0.0 in
        let batch_count = ref 0 in
        
        (* Process trajectories *)
        List.iter (fun traj ->
          let batches = BatchProcessing.create_batches 
            ~batch_size:config.batch_size traj in
          
          (* Process batches *)
          List.iter (fun batch ->
            let grads = process_batch model config.grad_estimator batch in
            
            (* Compute loss and update *)
            let loss = List.fold_left (fun acc grad ->
              acc +. Tensor.float_value (Tensor.mean grad)
            ) 0.0 grads in
            
            ExtendedOptimization.optimization_step 
              config.optimizer model.vs state.step;
            
            total_loss := !total_loss +. loss;
            batch_count := !batch_count + 1;
            state.step <- state.step + 1;
          ) batches;
        ) trajectories;
        
        (* Compute average loss *)
        let avg_loss = !total_loss /. float_of_int !batch_count in
        
        (* Early stopping check *)
        if avg_loss < state.best_loss then begin
          state.best_loss <- avg_loss;
          state.patience_counter <- 0;
        end else begin
          state.patience_counter <- state.patience_counter + 1;
          if state.patience_counter >= config.early_stopping_patience then
            state.should_stop <- true;
        end;
        
        (* Checkpointing *)
        if epoch mod config.checkpoint_interval = 0 then
          Printf.printf "Epoch %d: Average Loss = %f\n" epoch avg_loss;
        
        epoch_loop (epoch + 1)
      end
    in
    
    epoch_loop 0
end

(* Batch processing module *)
module BatchProcessing = struct
  type t = {
    batch_size: int;
    shuffle: bool;
    drop_last: bool;
  }

  let make ~batch_size ~shuffle ~drop_last =
    { batch_size; shuffle; drop_last }

  (* Shuffle trajectories while maintaining correspondence *)
  let shuffle_trajectories trajectories =
    let n = List.length trajectories in
    let indices = List.init n (fun i -> i) in
    let shuffled_indices = List.sort (fun _ _ -> 
      if Random.bool () then 1 else -1) indices in
    List.map (fun i -> List.nth trajectories i) shuffled_indices

  (* Create mini-batches respecting trajectory boundaries *)
  let create_batches t traj =
    let data, times, indices = traj in
    let n = List.length data in
    
    let rec make_batch acc curr_idx =
      if curr_idx >= n then List.rev acc
      else
        let end_idx = min (curr_idx + t.batch_size) n in
        if t.drop_last && end_idx - curr_idx < t.batch_size then
          List.rev acc
        else
          let batch_data = List.sub data curr_idx (end_idx - curr_idx) in
          let batch_times = List.sub times curr_idx (end_idx - curr_idx) in
          let batch_indices = List.sub indices curr_idx (end_idx - curr_idx) in
          make_batch ((batch_data, batch_times, batch_indices) :: acc) end_idx
    in
    make_batch [] 0

  (* Process batch with optional augmentation *)
  let process_batch t batch augmentation =
    let data, times, indices = batch in
    match augmentation with
    | None -> batch
    | Some aug -> DataAugmentation.apply_augmentation aug (data, times, indices)
end

(* Data augmentation *)
module DataAugmentation = struct
  type augmentation_type =
    | TimeShift of float
    | TimeScale of float
    | AdditiveSmoothNoise of float
    | TemporalMixup of float

  type t = {
    augmentations: augmentation_type list;
    probability: float;
  }

  let make ~augmentations ~probability = 
    { augmentations; probability }

  let apply_augmentation t trajectory =
    let data, times, indices = trajectory in
    
    List.fold_left (fun (curr_data, curr_times, curr_indices) aug ->
      if Random.float 1.0 > t.probability then
        (curr_data, curr_times, curr_indices)
      else
        match aug with
        | TimeShift delta ->
            let new_times = List.map (fun t -> t +. delta) curr_times in
            (curr_data, new_times, curr_indices)
        | TimeScale scale ->
            let new_times = List.map (fun t -> t *. scale) curr_times in
            (curr_data, new_times, curr_indices)
        | AdditiveSmoothNoise sigma ->
            let noise = List.map (fun x ->
              let eps = Tensor.randn_like x in
              Tensor.(x + (eps * float_of_int sigma))
            ) curr_data in
            (noise, curr_times, curr_indices)
        | TemporalMixup alpha ->
            let n = List.length curr_data in
            let shift = max 1 (int_of_float (float_of_int n *. alpha)) in
            let shifted_data = List.drop shift curr_data @ 
                             List.take shift curr_data in
            let mixed = List.map2 (fun x1 x2 ->
              let lam = Random.float 1.0 in
              Tensor.(x1 * float_of_int lam + x2 * float_of_int (1.0 -. lam))
            ) curr_data shifted_data in
            (mixed, curr_times, curr_indices)
    ) (data, times, indices) t.augmentations

end

(* Trajectory handling *)
module Trajectory = struct
  type t = {
    data: Tensor.t list;
    times: float list;
    batch_indices: int list;
  }

  let make data times batch_indices =
    { data; times; batch_indices }

  (* Split trajectory into batches *)
  let to_batches t batch_size =
    let n = List.length t.data in
    let rec split_batches acc i =
      if i >= n then List.rev acc
      else
        let end_idx = min (i + batch_size) n in
        let batch_data = List.sub t.data i (end_idx - i) in
        let batch_times = List.sub t.times i (end_idx - i) in
        let batch_indices = List.sub t.batch_indices i (end_idx - i) in
        split_batches ((batch_data, batch_times, batch_indices) :: acc) 
                     (i + batch_size)
    in
    split_batches [] 0

  (* Combine multiple trajectories *)
  let collate trajectories =
    let data = List.concat (List.map (fun t -> t.data) trajectories) in
    let times = List.concat (List.map (fun t -> t.times) trajectories) in
    let batch_indices = 
      List.concat (
        List.mapi (fun i t -> 
          List.init (List.length t.data) (fun _ -> i)
        ) trajectories
      ) in
    make data times batch_indices
end

(* Initialization *)
module Initialization = struct
  type init_method =
    | Xavier
    | KaimingNormal
    | Orthogonal
    | Custom of (Tensor.t -> Tensor.t)

  let init_weights method_ tensor =
    match method_ with
    | Xavier ->
        let fan_in, fan_out = 
          match Tensor.size tensor with
          | [|out_dim; in_dim|] -> in_dim, out_dim
          | _ -> failwith "Invalid tensor dimensions" in
        let std = sqrt (2.0 /. float_of_int (fan_in + fan_out)) in
        Tensor.(tensor * float_of_int std)
    | KaimingNormal ->
        let fan_in = 
          match Tensor.size tensor with
          | [|_; in_dim|] -> in_dim
          | _ -> failwith "Invalid tensor dimensions" in
        let std = sqrt (2.0 /. float_of_int fan_in) in
        Tensor.(tensor * float_of_int std)
    | Orthogonal ->
        let m, n = Tensor.size2 tensor in
        let q, r = Tensor.qr tensor in
        let d = Tensor.diagonal r ~dim1:0 ~dim2:1 in
        let ph = Tensor.sign d in
        Tensor.(matmul q (diag ph))
    | Custom f -> f tensor

  let init_network method_ vs =
    Var_store.all_vars vs
    |> List.iter (fun var ->
      let initialized = init_weights method_ (Var_store.tensor var) in
      Tensor.copy_ (Var_store.tensor var) initialized)
end

(* Main model interface *)
module LatentSDE = struct
  type model_config = {
    dim: int;
    latent_dim: int;
    hidden_dim: int;
    partition_size: int;
    num_samples: int;
  }

  type t = {
    config: model_config;
    sde: SDECore.t;
    mgp: MGPCore.t;
    encoder: Amortization.t;
    temporal: TemporalContext.t;
    vs: Var_store.t;
  }

  let make ~config ~device =
    let vs = Var_store.create ~name:"latent_sde" ~device () in
    
    let sde = SDECore.make {
      dim = config.dim;
      latent_dim = config.latent_dim;
      hidden_dim = config.hidden_dim;
      time_embedding_dim = config.hidden_dim;
    } in
    
    let mgp = MGPCore.make config.latent_dim config.hidden_dim in
    
    let encoder = Amortization.make
      ~partition_size:config.partition_size
      ~overlap:(config.partition_size / 4)
      ~latent_dim:config.latent_dim
      ~hidden_dim:config.hidden_dim in
    
    let temporal = TemporalContext.make
      ~window_size:config.partition_size
      ~stride:(config.partition_size / 2)
      ~kernel_size:3
      ~hidden_dim:config.hidden_dim
      ~attention_heads:4 in
    
    { config; sde; mgp; encoder; temporal; vs }

  (* Forward pass through the model *)
  let forward t x time =
    let encoded = t.encoder.encoder_net x in
    let temporal_features = t.temporal.process_sequence encoded time in
    let mean, cov = t.mgp.compute_derivatives temporal_features encoded time in
    let z = SDECore.sample_initial t.sde 1 in
    z, mean, cov

  (* Loss computation *)
  let compute_loss t x time =
    let z, mean, cov = forward t x time in
    let drift = t.sde.drift_net z time in
    let diff = Tensor.(drift - mean) in
    let loss = Tensor.(mean (diff * diff / cov)) in
    loss

  (* Training interface *)
  let train t config trajectories =
    let optimizer = ExtendedOptimization.make
      ~scheduler:(LinearWarmup {
        warmup_steps = 1000;
        peak_lr = 1e-3;
      })
      ~clip_grad_norm:(Some 1.0)
      ~weight_decay:1e-5
      ~ema_decay:(Some 0.999) in

    let grad_estimator = GradientEstimation.make
      ~outer_samples:100
      ~inner_samples:10
      ~stratification:true
      ~batch_size:config.batch_size in

    let trainer_config = {
      Trainer.epochs = config.epochs;
      batch_size = config.batch_size;
      partition_size = config.partition_size;
      optimizer;
      grad_estimator;
      checkpoint_interval = 100;
      early_stopping_patience = 10;
    } in

    Trainer.train trainer_config t trajectories
end

(* Utility functions *)
module Util = struct
  let tensor_to_list tensor =
    let size = Tensor.size1 tensor in
    List.init size (fun i -> Tensor.get tensor [|i|])

  let list_to_tensor lst =
    let arr = Array.of_list lst in
    Tensor.of_float1 arr

  let split_list lst n =
    let rec aux acc curr = function
      | [] -> List.rev (List.rev curr :: acc)
      | h :: t when List.length curr = n -> 
          aux (List.rev curr :: acc) [h] t
      | h :: t -> aux acc (h :: curr) t
    in
    aux [] [] lst

  let pad_list lst n default =
    if List.length lst >= n then lst
    else
      lst @ List.init (n - List.length lst) (fun _ -> default)

  let sliding_window lst window_size stride =
    let n = List.length lst in
    let rec aux acc i =
      if i >= n then List.rev acc
      else
        let window = List.sub lst i (min window_size (n - i)) in
        aux (window :: acc) (i + stride)
    in
    aux [] 0
end