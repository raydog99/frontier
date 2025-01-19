open Torch

module ComplexOps = struct
  type t = {re: Tensor.t; im: Tensor.t}
  
  let eps = 1e-10

  let complex_mul a b = {
    re = Tensor.(a.re * b.re - a.im * b.im);
    im = Tensor.(a.re * b.im + a.im * b.re)
  }

  let complex_add a b = {
    re = Tensor.(a.re + b.re);
    im = Tensor.(a.im + b.im)
  }

  let complex_exp {re; im} =
    let magnitude = Tensor.exp re in
    {
      re = Tensor.(magnitude * cos im);
      im = Tensor.(magnitude * sin im)
    }

  let to_complex t = {re = t; im = Tensor.zeros_like t}

  let complex_matrix_mul a b = {
    re = Tensor.(mm a.re b.re - mm a.im b.im);
    im = Tensor.(mm a.re b.im + mm a.im b.re)
  }

  let safe_log x = Tensor.(log (x + f eps))
  let safe_div num denom = Tensor.(num / (denom + f eps))
end

type event = {
    time: float;
    mark: int;
    index: int;
  }

type history = {
    events: event array;
    times: Tensor.t;
    marks: Tensor.t;
    count: int;
  }

type model_config = {
    hidden_dim: int;          (* State dimension P *)
    mark_dim: int;           (* Number of marks K *)
    embedding_dim: int;      (* Embedding dimension R *)
    num_layers: int;         (* Number of LLH layers L *)
    use_input_dependent: bool; (* Whether to use input-dependent dynamics *)
    max_memory: int;         (* Maximum memory for state caching *)
  }

type layer_state = {
    x: ComplexOps.t;        (* Hidden state *)
    u: Tensor.t;           (* Input signal *)
    y: Tensor.t;           (* Output signal *)
    time: float;           (* Current time *)
  }

let empty_layer_state = {
    x = {re = Tensor.zeros [1]; im = Tensor.zeros [1]};
    u = Tensor.zeros [1];
    y = Tensor.zeros [1];
    time = 0.0;
}

(* Numerical stability and eigenvalue computation *)
let stabilize_eigenvalues tensor =
    let real = Tensor.real tensor in
    let imag = Tensor.imag tensor in
    let stable_real = Tensor.(neg (abs real + f eps)) in
    {re = stable_real; im = imag}

let compute_condition_number v v_inv =
    let norm_v = Tensor.(norm v ~p:2 + f eps) in
    let norm_v_inv = Tensor.(norm v_inv ~p:2 + f eps) in
    Tensor.to_float0_exn Tensor.(norm_v * norm_v_inv)

(* QR decomposition with stability *)
let qr_decomposition a =
    let n = Tensor.shape a |> List.hd in
    let q = Tensor.eye n in
    let r = Tensor.copy a in
    
    for i = 0 to n - 2 do
      let x = Tensor.slice r [i] [n] [i] [i+1] in
      let norm = Tensor.(norm x ~p:2 + f eps) in
      if Tensor.to_float0_exn norm > eps then begin
        let u = Tensor.slice r [i] [n] [i] [i+1] in
        let u = Tensor.(copy_sub u [0] (neg norm)) in
        let u_norm = Tensor.(norm u ~p:2 + f eps) in
        if Tensor.to_float0_exn u_norm > eps then begin
          let u = Tensor.(u / u_norm) in
          let h = Tensor.(eye (n-i) - (mm (reshape u [n-i; 1]) 
                                        (reshape u [1; n-i])) * f 2.) in
          let q_update = Tensor.(mm (slice q [0] [n] [i] [n]) h) in
          let r_update = Tensor.(mm h (slice r [i] [n] [i] [n])) in
          Tensor.copy_ (Tensor.slice q [0] [n] [i] [n]) q_update;
          Tensor.copy_ (Tensor.slice r [i] [n] [i] [n]) r_update
        end
      end
    done;
    (q, r)

(* Complex eigenvalue system handling *)
module ComplexEigenSystem = struct
  type eigen_system = {
    eigenvalues: complex array;
    v: complex;
    v_inv: complex;
    condition_number: float;
  }

  let balance_matrix a =
    let n = Tensor.shape a |> List.hd in
    let d = Tensor.ones [n] in
    
    for _ = 1 to 10 do
      for i = 0 to n - 1 do
        let row_norm = Tensor.norm (Tensor.select a 0 i) ~p:1 in
        let col_norm = Tensor.norm (Tensor.select a 1 i) ~p:1 in
        let f = sqrt (Tensor.to_float0_exn Tensor.(row_norm / col_norm)) in
        Tensor.mul_ (Tensor.select a 0 i) (Tensor.f (1. /. f));
        Tensor.mul_ (Tensor.select a 1 i) (Tensor.f f);
        d.[[i]] <- Tensor.f f
      done
    done;
    (a, d)

  let compute_eigensystem a =
    let balanced_a, d = balance_matrix {re = a; im = Tensor.zeros_like a} in
    let n = Tensor.shape a |> List.hd in
    
    (* Compute Schur decomposition *)
    let rec schur_iterate mat iter max_iter tol =
      if iter >= max_iter then mat
      else
        let q, r = StableEigen.qr_decomposition mat.re in
        let new_a = {
          re = Tensor.(mm r q);
          im = Tensor.zeros_like mat.re
        } in
        let diff = Tensor.(abs (new_a.re - mat.re)) in
        if Tensor.to_float0_exn (Tensor.max diff) < tol then new_a
        else schur_iterate new_a (iter + 1) max_iter tol
    in
    
    let schur_form = schur_iterate balanced_a 0 100 1e-10 in
    
    (* Extract eigenvalues and eigenvectors *)
    let eigenvals = Array.init n (fun i ->
      {
        re = Tensor.get schur_form.re [i; i];
        im = if i < n - 1 then
               Tensor.get schur_form.re [i; i+1]
             else Tensor.zeros [1]
      }
    ) in
    
    (* Stabilize eigenvalues *)
    let stable_eigenvals = Array.map (fun ev ->
      let magnitude = sqrt (Tensor.to_float0_exn (
        Tensor.(square ev.re + square ev.im))
      ) in
      let stabilized_re = Tensor.(
        ev.re * f (min 1. (1. /. magnitude)) * f (-1.)
      ) in
      {re = stabilized_re; im = ev.im}
    ) eigenvals in
    
    (* Compute eigenvectors *)
    let v_mat = Tensor.eye n in
    let v_inv = Tensor.inverse v_mat in
    
    {
      eigenvalues = stable_eigenvals;
      v = {re = v_mat; im = Tensor.zeros_like v_mat};
      v_inv = {re = v_inv; im = Tensor.zeros_like v_inv};
      condition_number = StableEigen.compute_condition_number v_mat v_inv;
    }
end

(* Stochastic jump differential equations *)
module SJDE = struct
  type jump_process = {
    intensity: Tensor.t;
    mark_effect: Tensor.t;
    time: float;
  }

  let compute_jump_effect config prev_state jump =
    let time_diff = Tensor.f (jump.time -. prev_state.time) in
    let decay = Tensor.exp (Tensor.(neg time_diff)) in
    Tensor.(jump.mark_effect * decay)

  let integrate_jumps config state jumps current_time =
    Array.fold_left (fun acc jump ->
      if jump.time >= current_time then acc
      else
        let effect = compute_jump_effect config state jump in
        Tensor.(acc + effect)
    ) (Tensor.zeros [config.hidden_dim]) jumps

  let solve_sjde layer state jumps dt =
    (* Compute continuous evolution *)
    let continuous_evolution = 
      complex_exp (complex_mul layer.eigenvals (to_complex (Tensor.f dt))) in
    
    (* Add jump effects *)
    let jump_effects = integrate_jumps layer.config state jumps state.time in
    let new_state = {
      x = complex_add (complex_mul continuous_evolution state.x) 
          (to_complex jump_effects);
      u = state.u;
      y = state.y;
      time = state.time +. dt
    } in
    new_state
end

(* Memory efficient state management *)
module StateManagement = struct
  type compression_config = {
    compression_ratio: int;
    max_memory: int;
    interpolation_points: int;
  }

  type state_cache = {
    states: (int, layer_state) Hashtbl.t;
    times: float array;
    max_size: int;
  }

  let create_cache max_size = {
    states = Hashtbl.create max_size;
    times = [||];
    max_size;
  }

  let compress_state state config =
    let n = Tensor.shape state.x.re |> List.hd in
    let reduced_dim = n / config.compression_ratio in
    let projection = Tensor.randn [reduced_dim; n] in
    {
      x = {
        re = Tensor.(mm projection state.x.re);
        im = Tensor.(mm projection state.x.im);
      };
      u = Tensor.(mm projection state.u);
      y = Tensor.(mm projection state.y);
      time = state.time;
    }

  let interpolate_states s1 s2 t =
    let alpha = (t -. s1.time) /. (s2.time -. s1.time) in
    {
      x = {
        re = Tensor.(s1.x.re * f (1. -. alpha) + s2.x.re * f alpha);
        im = Tensor.(s1.x.im * f (1. -. alpha) + s2.x.im * f alpha);
      };
      u = Tensor.(s1.u * f (1. -. alpha) + s2.u * f alpha);
      y = Tensor.(s1.y * f (1. -. alpha) + s2.y * f alpha);
      time = t;
    }

  let get_state_at_time cache t =
    let rec find_interval times idx =
      if idx >= Array.length times - 1 then idx - 1
      else if times.(idx) <= t && t < times.(idx + 1) then idx
      else find_interval times (idx + 1)
    in
    let idx = find_interval cache.times 0 in
    let s1 = Hashtbl.find cache.states idx in
    let s2 = Hashtbl.find cache.states (idx + 1) in
    interpolate_states s1 s2 t

  let update_cache cache state =
    if Hashtbl.length cache.states >= cache.max_size then begin
      (* Remove oldest state *)
      let oldest_time = cache.times.(0) in
      Hashtbl.remove cache.states (Array.length cache.times - 1);
      cache.times <- Array.sub cache.times 1 (Array.length cache.times - 1)
    end;
    let idx = Array.length cache.times in
    Hashtbl.add cache.states idx state;
    cache.times <- Array.append cache.times [|state.time|]
end

(* Layer normalization *)
module LayerNorm = struct
  type t = {
    gamma: Tensor.t;
    beta: Tensor.t;
    eps: float;
  }

  let create dim =
    {
      gamma = Tensor.ones [dim];
      beta = Tensor.zeros [dim];
      eps = 1e-5;
    }

  let forward ln x =
    let mean = Tensor.mean_dim x ~dim:[1] ~keepdim:true in
    let var = Tensor.var x ~dim:[1] ~unbiased:false ~keepdim:true in
    Tensor.(
      (ln.gamma * (x - mean) / sqrt(var + f ln.eps)) + ln.beta
    )
end

(* Mark embedding handling *)
module MarkEmbedding = struct
  type t = {
    weight: Tensor.t;
    dim: int;
    num_marks: int;
  }

  let create dim num_marks =
    {
      weight = Tensor.randn [num_marks; dim];
      dim;
      num_marks;
    }

  let forward emb mark =
    Tensor.index_select emb.weight ~dim:0 ~index:mark

  let compute_attention emb queries keys values scale =
    let scores = Tensor.(mm queries (transpose keys ~dim0:0 ~dim1:1)) in
    let scaled_scores = Tensor.(f scale * scores) in
    let weights = Tensor.softmax scaled_scores ~dim:(-1) in
    Tensor.(mm weights values)
end

(* Latent Linear Hawkes Layer *)
module LLH_Layer = struct
  type t = {
    a: complex;              (* State transition *)
    b: Tensor.t;            (* Input projection *)
    e: Tensor.t;            (* Event embedding *)
    c: Tensor.t;            (* Output projection *)
    d: Tensor.t;            (* Direct feedthrough *)
    eigenvals: complex;      (* Cached eigenvalues *)
    v: complex;             (* Eigenvectors *)
    v_inv: complex;         (* Inverse eigenvectors *)
    config: model_config;    (* Layer configuration *)
  }

  let create config =
    let dim = config.hidden_dim in
    let a = {
      re = Tensor.randn [dim; dim];
      im = Tensor.randn [dim; dim];
    } in
    let eigen = ComplexEigenSystem.compute_eigensystem a.re in
    {
      a;
      b = Tensor.randn [dim; config.embedding_dim];
      e = Tensor.randn [dim; config.mark_dim];
      c = Tensor.randn [config.embedding_dim; dim];
      d = Tensor.randn [config.embedding_dim; config.embedding_dim];
      eigenvals = eigen.eigenvalues.(0);
      v = eigen.v;
      v_inv = eigen.v_inv;
      config;
    }

  let discretize layer dt =
    (* Compute matrix exponential for state transition *)
    let exp_lambda = complex_exp (
      complex_mul layer.eigenvals (to_complex (Tensor.f dt))
    ) in
    {layer with a = exp_lambda}

  let forward layer state input =
    (* State evolution *)
    let evolved_state = complex_mul layer.a state.x in
    (* Input processing *)
    let input_effect = Tensor.(mm layer.b input) in
    (* Combine effects *)
    let new_state = {
      x = complex_add evolved_state (to_complex input_effect);
      u = input;
      y = Tensor.(mm layer.c state.x.re + mm layer.d input);
      time = state.time;
    } in
    new_state

  let forward_with_jumps layer state time mark =
    let dt = time -. state.time in
    let discretized = discretize layer dt in
    let base_state = forward discretized state (Tensor.zeros [layer.config.embedding_dim]) in
    
    (* Add mark-specific jump *)
    let jump_effect = Tensor.(mm layer.e (Tensor.one_hot mark layer.config.mark_dim)) in
    {base_state with
      x = complex_add base_state.x (to_complex jump_effect);
      time
    }
end

(* Linear Hawkes Process model *)
module LHP = struct
  type t = {
    layers: LLH_Layer.t array;
    mark_embedding: MarkEmbedding.t;
    layer_norms: LayerNorm.t array;
    output_projection: Tensor.t;
    output_bias: Tensor.t;
    scale: Tensor.t;
    config: model_config;
    state_cache: StateManagement.state_cache;
  }

  let create config =
    {
      layers = Array.init config.num_layers (fun _ -> LLH_Layer.create config);
      mark_embedding = MarkEmbedding.create config.embedding_dim config.mark_dim;
      layer_norms = Array.init config.num_layers (fun _ -> LayerNorm.create config.hidden_dim);
      output_projection = Tensor.randn [config.mark_dim; config.hidden_dim];
      output_bias = Tensor.zeros [config.mark_dim];
      scale = Tensor.ones [config.mark_dim];
      config;
      state_cache = StateManagement.create_cache config.max_memory;
    }

  let forward_layer model layer_idx state time =
    let layer = model.layers.(layer_idx) in
    let norm = model.layer_norms.(layer_idx) in
    let state' = LLH_Layer.forward layer state (Tensor.zeros [model.config.hidden_dim]) in
    {state' with y = LayerNorm.forward norm state'.y}

  let compute_intensity model state time =
    (* Project final state to intensity space *)
    let projected = Tensor.(mm model.output_projection state.y + model.output_bias) in
    (* Apply softplus and scaling *)
    Tensor.(model.scale * softplus projected)
end

(* Training and optimization *)
module Training = struct
  type training_config = {
    batch_size: int;
    learning_rate: float;
    max_epochs: int;
    patience: int;
    grad_clip: float;
    weight_decay: float;
    scheduler_factor: float;
    min_lr: float;
    num_monte_carlo: int;
  }

  type optimizer_state = {
    momentum: (string * Tensor.t) list;
    velocity: (string * Tensor.t) list;
    beta1: float;
    beta2: float;
    eps: float;
    step: int;
  }

  let create_optimizer_state params =
    {
      momentum = List.map (fun (name, p) -> (name, Tensor.zeros_like p)) params;
      velocity = List.map (fun (name, p) -> (name, Tensor.zeros_like p)) params;
      beta1 = 0.9;
      beta2 = 0.999;
      eps = 1e-8;
      step = 0;
    }

  let collect_parameters model =
    let layer_params = Array.fold_left (fun acc layer ->
      let params = [
        ("state_matrix", layer.LLH_Layer.a.re);
        ("input_matrix", layer.b);
        ("embedding_matrix", layer.e);
        ("output_matrix", layer.c);
        ("direct_matrix", layer.d);
      ] in
      List.append acc params
    ) [] model.DLHP.layers in
    
    let other_params = [
      ("mark_embedding", model.mark_embedding.MarkEmbedding.weight);
      ("output_projection", model.output_projection);
      ("output_bias", model.output_bias);
      ("scale", model.scale);
    ] in
    
    List.append layer_params other_params

  let adam_update params opt_state lr =
    let step = opt_state.step + 1 in
    let beta1_t = opt_state.beta1 ** float_of_int step in
    let beta2_t = opt_state.beta2 ** float_of_int step in
    
    let updated_params = List.map2 (fun (name, param) (_, m, v) ->
      let grad = match Tensor.grad param with
        | Some g -> g
        | None -> Tensor.zeros_like param
      in
      
      let m' = Tensor.(
        f opt_state.beta1 * m + f (1. -. opt_state.beta1) * grad
      ) in
      let v' = Tensor.(
        f opt_state.beta2 * v + f (1. -. opt_state.beta2) * (grad * grad)
      ) in
      
      let m_hat = Tensor.(m' / (f (1. -. beta1_t))) in
      let v_hat = Tensor.(v' / (f (1. -. beta2_t))) in
      
      let update = Tensor.(
        m_hat / (sqrt v_hat + f opt_state.eps) * f lr
      ) in
      
      Tensor.(param -= update);
      (name, m', v')
    ) params (List.combine3 
      (List.map fst opt_state.momentum)
      (List.map snd opt_state.momentum)
      (List.map snd opt_state.velocity))
    in
    
    let new_momentum, new_velocity = List.split (List.map (fun (n, m, v) -> 
      ((n, m), (n, v))) updated_params) in
    
    {opt_state with 
      momentum = new_momentum;
      velocity = new_velocity;
      step
    }

  let clip_gradients params clip_value =
    let total_norm = List.fold_left (fun acc (_, p) ->
      match Tensor.grad p with
      | Some g -> 
          let norm = Tensor.norm g ~p:2 in
          acc +. (Tensor.to_float0_exn norm) ** 2.
      | None -> acc
    ) 0. params in
    
    let total_norm = sqrt total_norm in
    if total_norm > clip_value then
      let scale = clip_value /. total_norm in
      List.iter (fun (_, p) ->
        match Tensor.grad p with
        | Some g -> Tensor.(g *= f scale)
        | None -> ()
      ) params

  let compute_loss model history config =
    (* Event log-likelihood terms *)
    let event_ll = Array.fold_left (fun acc event ->
      let state = StateManagement.get_state_at_time model.state_cache event.time in
      let intensity = DLHP.compute_intensity model state event.time in
      let mark_intensity = Tensor.get intensity [event.mark] in
      acc +. (Tensor.to_float0_exn (safe_log mark_intensity))
    ) 0. history.events in
    
    (* Monte Carlo approximation of the integral term *)
    let total_time = history.events.(Array.length history.events - 1).time in
    let sample_times = Array.init config.num_monte_carlo 
      (fun _ -> Random.float total_time) in
    let integral_term = Array.fold_left (fun acc time ->
      let state = StateManagement.get_state_at_time model.state_cache time in
      let intensity = DLHP.compute_intensity model state time in
      acc +. (Tensor.to_float0_exn (Tensor.sum intensity)) /. 
        float_of_int config.num_monte_carlo
    ) 0. sample_times in
    
    event_ll -. integral_term
end

(* Batch processing and training loops *)
module BatchProcessor = struct
  type batch = {
    states: layer_state array array;  (* [batch_size][num_layers] *)
    times: float array;
    marks: int array;
    masks: Tensor.t;
    memory_config: StateManagement.compression_config;
  }

  let create_batch size config =
    {
      states = Array.make_matrix size config.num_layers empty_layer_state;
      times = Array.make size 0.;
      marks = Array.make size 0;
      masks = Tensor.ones [size];
      memory_config = {
        compression_ratio = 4;
        interpolation_points = 100;
        max_memory = 1000;
      };
    }

  let create_batches history config =
    let n = Array.length history.events in
    let num_batches = (n + config.batch_size - 1) / config.batch_size in
    
    Array.init num_batches (fun i ->
      let start_idx = i * config.batch_size in
      let end_idx = min (start_idx + config.batch_size) n in
      let batch_size = end_idx - start_idx in
      
      let batch = create_batch batch_size config in
      
      for j = 0 to batch_size - 1 do
        let event = history.events.(start_idx + j) in
        batch.times.(j) <- event.time;
        batch.marks.(j) <- event.mark;
      done;
      
      if batch_size < config.batch_size then
        batch.masks <- Tensor.cat [
          Tensor.ones [batch_size];
          Tensor.zeros [config.batch_size - batch_size]
        ] ~dim:0;
      
      batch
    )

  let process_batch model batch =
    let batch_size = Array.length batch.states in
    Array.init batch_size (fun b ->
      if Tensor.to_float0_exn batch.masks.[[b]] > 0. then begin
        Array.mapi (fun l state ->
          DLHP.forward_layer model l state batch.times.(b)
        ) batch.states.(b)
      end else
        batch.states.(b)
    )
end

(* Training loop *)
module TrainingLoop = struct
  type training_state = {
    epoch: int;
    best_loss: float;
    patience_counter: int;
    current_lr: float;
    optimizer_state: optimizer_state;
  }

  let create_training_state config params =
    {
      epoch = 0;
      best_loss = Float.infinity;
      patience_counter = 0;
      current_lr = config.learning_rate;
      optimizer_state = create_optimizer_state params;
    }

  let update_learning_rate state config loss =
    if loss < state.best_loss then
      (state.current_lr, 0)  (* Reset patience if loss improved *)
    else begin
      let patience_counter = state.patience_counter + 1 in
      if patience_counter >= config.patience then
        (max config.min_lr (state.current_lr *. config.scheduler_factor), 0)
      else
        (state.current_lr, patience_counter)
    end

  let train_epoch model config state data =
    let batches = BatchProcessor.create_batches data config.batch_size in
    
    Array.fold_left (fun (state, total_loss) batch ->
      (* Forward pass *)
      let states = BatchProcessor.process_batch model batch in
      let loss = compute_loss model {
        events = Array.mapi (fun i _ -> 
          {time = batch.times.(i); mark = batch.marks.(i); index = i}
        ) batch.times;
        times = Tensor.of_float1 batch.times;
        marks = Tensor.of_int1 batch.marks;
        count = Array.length batch.times;
      } config in
      
      (* Backward pass *)
      Tensor.backward loss;
      
      (* Gradient clipping *)
      let params = collect_parameters model in
      if config.grad_clip > 0. then
        clip_gradients params config.grad_clip;
      
      (* Optimization step *)
      let opt_state = adam_update params state.optimizer_state state.current_lr in
      
      (* Update learning rate *)
      let new_lr, new_patience = update_learning_rate state config 
        (Tensor.to_float0_exn loss) in
      
      let new_state = {
        state with
        optimizer_state = opt_state;
        current_lr = new_lr;
        patience_counter = new_patience;
        best_loss = min state.best_loss (Tensor.to_float0_exn loss);
      } in
      
      (new_state, total_loss +. Tensor.to_float0_exn loss)
    ) (state, 0.) batches
end

(* Evaluation and utility *)
module Evaluation = struct
  type metrics = {
    log_likelihood: float;
    mean_intensity: float;
    mark_distribution: Tensor.t;
  }

  type prediction = {
    time: float;
    mark: int;
    intensity: Tensor.t;
    probability: float;
  }

  let compute_metrics model history =
    (* Compute log-likelihood *)
    let ll = Training.compute_loss model history {
      Training.num_monte_carlo = 1000;
      batch_size = 1;
      learning_rate = 0.;
      max_epochs = 0;
      patience = 0;
      grad_clip = 0.;
      weight_decay = 0.;
      scheduler_factor = 0.;
      min_lr = 0.;
    } in

    (* Compute mean intensity *)
    let total_time = history.events.(Array.length history.events - 1).time in
    let sample_points = 1000 in
    let dt = total_time /. float_of_int sample_points in
    let mean_intensity = Array.init sample_points (fun i ->
      let t = float_of_int i *. dt in
      let state = StateManagement.get_state_at_time model.DLHP.state_cache t in
      let intensity = DLHP.compute_intensity model state t in
      Tensor.to_float0_exn (Tensor.mean intensity)
    ) |> Array.fold_left (+.) 0. |> fun x -> x /. float_of_int sample_points in

    (* Compute mark distribution *)
    let mark_counts = Array.make model.config.mark_dim 0 in
    Array.iter (fun event ->
      mark_counts.(event.mark) <- mark_counts.(event.mark) + 1
    ) history.events;
    let total_events = float_of_int (Array.length history.events) in
    let mark_dist = Tensor.of_float1 (Array.map (fun count ->
      float_of_int count /. total_events
    ) mark_counts) in

    {
      log_likelihood = ll;
      mean_intensity;
      mark_distribution = mark_dist;
    }

  let predict_next_event model history current_time =
    let state = StateManagement.get_state_at_time model.DLHP.state_cache current_time in
    let intensity = DLHP.compute_intensity model state current_time in
    
    (* Sample next event time using thinning algorithm *)
    let rec sample_time upper_bound t =
      let proposed_time = t +. Random.float upper_bound in
      let proposed_state = StateManagement.get_state_at_time 
        model.DLHP.state_cache proposed_time in
      let proposed_intensity = DLHP.compute_intensity model proposed_state proposed_time in
      let total_intensity = Tensor.to_float0_exn (Tensor.sum proposed_intensity) in
      
      if Random.float 1.0 <= total_intensity /. upper_bound then
        (proposed_time, proposed_intensity)
      else
        sample_time upper_bound proposed_time
    in
    
    let upper_intensity = Tensor.to_float0_exn (Tensor.max intensity) *. 1.2 in
    let (next_time, next_intensity) = sample_time upper_intensity current_time in
    
    (* Sample mark type *)
    let mark_probs = Tensor.softmax next_intensity ~dim:0 in
    let mark = sample_categorical mark_probs in
    
    {
      time = next_time;
      mark;
      intensity = next_intensity;
      probability = Tensor.to_float0_exn (Tensor.get mark_probs [mark]);
    }

  let sample_categorical probs =
    let cumsum = Array.make (Tensor.shape probs |> List.hd) 0. in
    let _ = Array.fold_left (fun acc i ->
      let p = Tensor.to_float0_exn (Tensor.get probs [i]) in
      cumsum.(i) <- acc +. p;
      acc +. p
    ) 0. (Array.init (Array.length cumsum) (fun i -> i)) in
    
    let r = Random.float 1.0 in
    let rec find_index idx =
      if idx >= Array.length cumsum then Array.length cumsum - 1
      else if r <= cumsum.(idx) then idx
      else find_index (idx + 1)
    in
    find_index 0
end

(* Visualization and logging *)
module Visualization = struct
  type plot_config = {
    num_points: int;
    time_range: float * float;
    mark_colors: string array;
  }

  (* Create intensity plot data *)
  let create_intensity_plot_data model history config =
    let (start_time, end_time) = config.time_range in
    let dt = (end_time -. start_time) /. float_of_int config.num_points in
    
    Array.init config.num_points (fun i ->
      let t = start_time +. float_of_int i *. dt in
      let state = StateManagement.get_state_at_time model.DLHP.state_cache t in
      let intensity = DLHP.compute_intensity model state t in
      (t, intensity)
    )

  (* Create mark embedding visualization data *)
  let create_mark_embedding_viz model =
    let embeddings = model.DLHP.mark_embedding.MarkEmbedding.weight in
    let num_marks = model.config.mark_dim in
    
    (* Compute pairwise distances *)
    let distances = Array.make_matrix num_marks num_marks 0. in
    for i = 0 to num_marks - 1 do
      for j = 0 to num_marks - 1 do
        let emb_i = Tensor.select embeddings 0 i in
        let emb_j = Tensor.select embeddings 0 j in
        let diff = Tensor.(emb_i - emb_j) in
        distances.(i).(j) <- Tensor.to_float0_exn (Tensor.norm diff ~p:2)
      done
    done;
    distances
end

(* Logging utilities *)
module Logging = struct
  type log_entry = {
    epoch: int;
    train_loss: float;
    val_loss: float option;
    learning_rate: float;
    metrics: Evaluation.metrics;
    timestamp: float;
  }

  type training_log = {
    model_config: Types.model_config;
    entries: log_entry list;
    total_time: float;
  }

  let create_log_entry epoch train_loss val_loss lr metrics =
    {
      epoch;
      train_loss;
      val_loss;
      learning_rate = lr;
      metrics;
      timestamp = Unix.gettimeofday ();
    }

  let save_log log filename =
    let oc = open_out filename in
    (* Write model configuration *)
    Printf.fprintf oc "Model Configuration:\n";
    Printf.fprintf oc "Hidden dimension: %d\n" log.model_config.hidden_dim;
    Printf.fprintf oc "Mark dimension: %d\n" log.model_config.mark_dim;
    Printf.fprintf oc "Number of layers: %d\n" log.model_config.num_layers;
    Printf.fprintf oc "\nTraining Log:\n";
    
    (* Write log entries *)
    List.iter (fun entry ->
      Printf.fprintf oc "Epoch %d:\n" entry.epoch;
      Printf.fprintf oc "  Train Loss: %f\n" entry.train_loss;
      (match entry.val_loss with
      | Some vl -> Printf.fprintf oc "  Validation Loss: %f\n" vl
      | None -> ());
      Printf.fprintf oc "  Learning Rate: %f\n" entry.learning_rate;
      Printf.fprintf oc "  Log-Likelihood: %f\n" entry.metrics.log_likelihood;
      Printf.fprintf oc "  Mean Intensity: %f\n" entry.metrics.mean_intensity;
      Printf.fprintf oc "\n"
    ) log.entries;
    
    Printf.fprintf oc "\nTotal Training Time: %f seconds\n" log.total_time;
    close_out oc

  let print_progress entry =
    Printf.printf "\rEpoch %d - Loss: %f - LR: %f%!" 
      entry.epoch entry.train_loss entry.learning_rate;
    match entry.val_loss with
    | Some vl -> Printf.printf " - Val Loss: %f%!" vl
    | None -> ();
    Printf.printf "\n"

  let save_model model filename =
    let oc = open_out_bin filename in
    Marshal.to_channel oc model [];
    close_out oc

  let load_model filename =
    let ic = open_in_bin filename in
    let model = (Marshal.from_channel ic : DLHP.t) in
    close_in ic;
    model
end