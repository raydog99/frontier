open Torch

type observation = {
    volatility: float;
    log_return: float;
  }

type garch_params = {
    omega: float;
    alpha: float;
    beta: float;
  }

type activation =
    | Tanh
    | ReLU
    | Identity

type layer = {
    weights: Tensor.t;
    bias: Tensor.t;
    activation: activation;
  }

type fnn_config = {
    input_dim: int;
    hidden_dims: int list;
    output_dim: int;
  }

type fnn = {
    layers: layer list;
    config: fnn_config;
  }

type loss =
    | MSE
    | MAE
    | RMSE

type training_config = {
    max_epochs: int;
    batch_size: int;
    learning_rate: float;
    momentum: float;
    early_stopping_patience: int;
    loss_fn: loss;
  }

type training_state = {
    epoch: int;
    loss: float;
    best_loss: float;
    patience_counter: int;
  }

let eps = 1e-10
let max_exp = 20.0

let safe_log x =
  let open Torch in
  Tensor.log (Tensor.add x (Scalar.float eps))

let safe_div num denom =
  let open Torch in
  let safe_denom = Tensor.add denom (Scalar.float eps) in
  Tensor.div num safe_denom

let safe_exp x =
  let open Torch in
  let clipped = Tensor.clamp x
    ~min:(Scalar.float (-.max_exp))
    ~max:(Scalar.float max_exp) in
  Tensor.exp clipped

let stable_softmax x =
  let open Torch in
  let max_x = Tensor.max x ~dim:0 ~keepdim:true |> fst in
  let shifted = Tensor.sub x max_x in
  let exp_x = Tensor.exp shifted in
  let sum_exp = Tensor.sum exp_x ~dim:0 ~keepdim:true in
  safe_div exp_x sum_exp

let validate_tensor ?(name="tensor") t =
  let open Torch in
  if Tensor.isnan t |> Tensor.any |> Tensor.to_bool0_exn then
    Error (Printf.sprintf "NaN detected in %s" name)
  else if Tensor.isinf t |> Tensor.any |> Tensor.to_bool0_exn then
    Error (Printf.sprintf "Inf detected in %s" name)
  else
    Ok t

let calc_log_returns prices =
  let log_prices = Tensor.log prices in
  Tensor.diff log_prices ~dim:0 ~n:1

let calc_volatility returns window_size =
  let squared_returns = Tensor.mul returns returns in
  let running_sum = Tensor.sum_window squared_returns ~window:window_size in
  let volatility = Tensor.div running_sum (Scalar.float window_size) in
  Tensor.sqrt volatility

let calc_volatility_robust returns window_size =
  (* Handle missing data *)
  let clean_returns = 
    let mask = Tensor.isnan returns in
    let filled = Tensor.where returns
      ~condition:mask
      ~self:(Scalar.float 0.)
      ~other:returns in
    filled in
    
  (* Handle extreme outliers using median absolute deviation *)
  let med = Tensor.median clean_returns in
  let mad = Tensor.median (Tensor.abs (Tensor.sub clean_returns med)) in
  let threshold = Tensor.mul mad (Scalar.float 3.0) in
  
  let outlier_adjusted = Tensor.clamp clean_returns
    ~min:(Tensor.sub med threshold)
    ~max:(Tensor.add med threshold) in
    
  (* Standard volatility calculation on cleaned data *)
  let squared_returns = Tensor.mul outlier_adjusted outlier_adjusted in
  let running_sum = Tensor.sum_window squared_returns ~window:window_size in
  let volatility = Tensor.div running_sum (Scalar.float window_size) in
  Tensor.sqrt volatility

let create_windows data window_size =
  let rec make_windows acc remaining =
    if List.length remaining < window_size then
      List.rev acc
    else
      let window = List.take window_size remaining in
      make_windows (window :: acc) (List.tl remaining)
  in
  make_windows [] data

let data_to_tensor data =
  let open Torch in
  let n = List.length data in
  let tensor = Tensor.zeros [n; 2] in
  List.iteri (fun i obs ->
    Tensor.set_ tensor [i; 0] obs.volatility;
    Tensor.set_ tensor [i; 1] obs.log_return;
  ) data;
  tensor

let data_to_tensors data =
  let open Torch in
  let n = List.length data in
  let x = Tensor.zeros [n; 2] in
  let y = Tensor.zeros [n] in
  List.iteri (fun i obs ->
    Tensor.set_ x [i; 0] obs.volatility;
    Tensor.set_ x [i; 1] obs.log_return;
    Tensor.set_ y [i] obs.volatility;
  ) data;
  x, y

module FinancialData = struct
  type market_data = {
    open_prices: Tensor.t;
    high_prices: Tensor.t;
    low_prices: Tensor.t;
    close_prices: Tensor.t;
    volume: Tensor.t;
    timestamp: float array;
  }

  type market_features = {
    returns: Tensor.t;
    volatility: Tensor.t;
    ranges: Tensor.t;
    volume_ratio: Tensor.t;
  }

  let load_market_data file_path =
    let open Stdio in
    let rows = ref [] in
    let ic = In_channel.create file_path in
    let header = Option.get (In_channel.input_line ic) in
    let _ = String.split_on_char ',' header in (* Parse header *)
    
    let process_line line =
      match String.split_on_char ',' line with
      | [timestamp; open_; high; low; close; volume] ->
          let parse_float s = float_of_string (String.trim s) in
          (float_of_string timestamp,
           parse_float open_,
           parse_float high,
           parse_float low,
           parse_float close,
           parse_float volume)
      | _ -> failwith "Invalid CSV format"
    in
    
    let rec read_lines () =
      match In_channel.input_line ic with
      | Some line -> 
          rows := process_line line :: !rows;
          read_lines ()
      | None -> ()
    in
    read_lines ();
    In_channel.close ic;
    
    let data = List.rev !rows in
    let n = List.length data in
    
    let timestamps = Array.make n 0. in
    let opens = Tensor.zeros [n] in
    let highs = Tensor.zeros [n] in
    let lows = Tensor.zeros [n] in
    let closes = Tensor.zeros [n] in
    let volumes = Tensor.zeros [n] in
    
    List.iteri (fun i (ts, o, h, l, c, v) ->
      timestamps.(i) <- ts;
      Tensor.set_ opens [i] o;
      Tensor.set_ highs [i] h;
      Tensor.set_ lows [i] l;
      Tensor.set_ closes [i] c;
      Tensor.set_ volumes [i] v;
    ) data;
    
    {
      open_prices = opens;
      high_prices = highs;
      low_prices = lows;
      close_prices = closes;
      volume = volumes;
      timestamp = timestamps;
    }

  let calc_range_volatility {high_prices; low_prices; _} window_size =
    let ranges = Tensor.sub high_prices low_prices in
    let normalized_ranges = Tensor.div ranges low_prices in
    let sum_ranges = Tensor.sum_window normalized_ranges ~window:window_size in
    Tensor.sqrt (Tensor.div sum_ranges (Scalar.float window_size))

  let calc_volume_adjusted_volatility data volatility window_size =
    let volume_ma = Tensor.mean_window data.volume ~window:window_size in
    let volume_ratio = Tensor.div data.volume volume_ma in
    Tensor.mul volatility volume_ratio

  let calc_realized_volatility prices interval_minutes =
    let returns = calc_log_returns prices in
    let squared_returns = Tensor.mul returns returns in
    let n_intervals = 24 * 60 / interval_minutes in
    let daily_sum = Tensor.sum_window squared_returns ~window:n_intervals in
    Tensor.sqrt (Tensor.mul daily_sum (Scalar.float n_intervals))

  let create_features data window_size =
    let returns = calc_returns data.close_prices in
    let hist_vol = calc_volatility returns window_size in
    let range_vol = calc_range_volatility data window_size in
    let volume_ratio = calc_volume_adjusted_volatility data hist_vol window_size in
    
    {
      returns;
      volatility = hist_vol;
      ranges = range_vol;
      volume_ratio;
    }
end

module Garch = struct
  type param_bounds = {
    omega_bounds: float * float;
    alpha_bounds: float * float;
    beta_bounds: float * float;
  }

  let default_bounds = {
    omega_bounds = (1e-6, 1.0);
    alpha_bounds = (0., 0.3);
    beta_bounds = (0.6, 0.99);
  }

  let enforce_constraints params =
    let open Float in
    let omega = max (fst default_bounds.omega_bounds)
      (min (snd default_bounds.omega_bounds) params.omega) in
    let alpha = max (fst default_bounds.alpha_bounds)
      (min (snd default_bounds.alpha_bounds) params.alpha) in
    let beta = max (fst default_bounds.beta_bounds)
      (min (snd default_bounds.beta_bounds) params.beta) in
    
    (* Ensure α + β < 1 for stationarity *)
    let sum = alpha +. beta in
    if sum >= 1.0 then
      let scale = 0.99 /. sum in
      {omega; alpha = alpha *. scale; beta = beta *. scale}
    else
      {omega; alpha; beta}

  let predict params ~prev_volatility ~prev_return =
    let open Float in
    params.omega +. 
    params.alpha *. (prev_return ** 2.0) +.
    params.beta *. (prev_volatility ** 2.0)

  let predict_batch params data =
    let open Torch in
    let n = Tensor.size data 0 in
    let predictions = Tensor.zeros [n] in
    
    for i = 1 to n - 1 do
      let prev_vol = Tensor.get data [i-1; 0] |> Tensor.to_float0_exn in
      let prev_ret = Tensor.get data [i-1; 1] |> Tensor.to_float0_exn in
      let pred = predict params ~prev_volatility:prev_vol ~prev_return:prev_ret in
      Tensor.set_ predictions [i] pred
    done;
    
    predictions

  let compute_loss params data =
    let open Torch in
    let n = List.length data in
    let loss = ref 0. in
    
    List.iteri (fun i obs ->
      if i > 0 then
        let prev = List.nth data (i-1) in
        let pred = predict params 
          ~prev_volatility:prev.volatility 
          ~prev_return:prev.log_return in
        loss := !loss +. 
          ((obs.volatility -. pred) ** 2.)
    ) data;
    
    !loss /. float (n-1)

  let train_step params learning_rate data =
    let loss = compute_loss params data in
    let grad_omega = 1e-4 in  (* Approximate gradients *)
    let grad_alpha = 1e-4 in
    let grad_beta = 1e-4 in
    
    let new_params = {
      omega = params.omega -. learning_rate *. grad_omega;
      alpha = params.alpha -. learning_rate *. grad_alpha;
      beta = params.beta -. learning_rate *. grad_beta;
    } in
    
    enforce_constraints new_params

  let train data learning_rate max_iters =
    let rec train_loop iter params =
      if iter >= max_iters then params
      else
        let updated = train_step params learning_rate data in
        train_loop (iter + 1) updated
    in
    
    let initial_params = {omega = 0.1; alpha = 0.1; beta = 0.8} in
    train_loop 0 initial_params

  let train_with_constraints data learning_rate max_iters =
    let rec train_loop iter params =
      if iter >= max_iters then params
      else
        let updated = train_step params learning_rate data in
        let constrained = enforce_constraints updated in
        train_loop (iter + 1) constrained
    in
    
    let initial_params = {omega = 0.1; alpha = 0.1; beta = 0.8} in
    train_loop 0 initial_params
end

module FNN = struct
  let activate tensor = function
    | Tanh -> Tensor.tanh tensor
    | ReLU -> Tensor.relu tensor
    | Identity -> tensor

  let create_layer input_dim output_dim activation =
    let k = 1. /. Float.sqrt (float input_dim) in
    {
      weights = Tensor.uniform [output_dim; input_dim] ~low:(-.k) ~high:k;
      bias = Tensor.zeros [output_dim];
      activation;
    }

  let init_network {input_dim; hidden_dims; output_dim} =
    let rec build_layers in_dim dims acc =
      match dims with
      | [] -> 
          let output_layer = create_layer in_dim output_dim Identity in
          List.rev (output_layer :: acc)
      | d :: rest ->
          let layer = create_layer in_dim d Tanh in
          build_layers d rest (layer :: acc)
    in
    {
      layers = build_layers input_dim hidden_dims [];
      config = {input_dim; hidden_dims; output_dim};
    }

  let forward net input =
    List.fold_left (fun acc layer ->
      let output = Tensor.(mm layer.weights acc + layer.bias) in
      activate output layer.activation
    ) input net.layers

  let create_fnn2 () = 
    init_network {input_dim = 2; hidden_dims = [2]; output_dim = 1}

  let create_fnn3 () =
    init_network {input_dim = 2; hidden_dims = [3]; output_dim = 1}

  let create_fnn2_3 () =
    init_network {input_dim = 2; hidden_dims = [2; 3]; output_dim = 1}

  let train net config (input, target) =
    let rec train_epoch state =
      if state.epoch >= config.max_epochs || 
         state.patience_counter >= config.early_stopping_patience then
        state
      else
        (* Compute batch loss *)
        let pred = forward net input in
        let loss = match config.loss_fn with
          | MSE -> 
              let diff = Tensor.sub pred target in
              Tensor.mean (Tensor.mul diff diff)
          | MAE ->
              Tensor.mean (Tensor.abs (Tensor.sub pred target))
          | RMSE ->
              let diff = Tensor.sub pred target in
              Tensor.sqrt (Tensor.mean (Tensor.mul diff diff))
        in
        
        let loss_val = Tensor.to_float0_exn loss in
        
        (* Backpropagation *)
        List.iter (fun layer ->
          Tensor.backward layer.weights loss;
          Tensor.backward layer.bias loss;
        ) net.layers;
        
        (* Update parameters *)
        List.iter (fun layer ->
          let w_grad = Tensor.grad layer.weights in
          let b_grad = Tensor.grad layer.bias in
          
          let w_update = Tensor.mul w_grad (Scalar.float config.learning_rate) in
          let _ = Tensor.sub_ layer.weights w_update in
          
          let b_update = Tensor.mul b_grad (Scalar.float config.learning_rate) in
          let _ = Tensor.sub_ layer.bias b_update in
          ()
        ) net.layers;
        
        (* Update state *)
        let new_state = 
          if loss_val < state.best_loss then
            {epoch = state.epoch + 1;
             loss = loss_val;
             best_loss = loss_val;
             patience_counter = 0}
          else
            {epoch = state.epoch + 1;
             loss = loss_val;
             best_loss = state.best_loss;
             patience_counter = state.patience_counter + 1}
        in
        train_epoch new_state
    in
    
    train_epoch {
      epoch = 0;
      loss = Float.infinity;
      best_loss = Float.infinity;
      patience_counter = 0;
    }
end

module PMC = struct
  type state = {
    id: int;
    model: [ `GARCH of garch_params | `FNN of fnn ]
  }

  type t = {
    states: state array;
    transition_probs: Tensor.t;
    initial_probs: Tensor.t;
  }

  module StateConstraints = struct
    type constraint_type =
      | NonNegative
      | Sumto1
      | Range of float * float
      | Custom of (Tensor.t -> bool)

    let apply_constraints tensor = function
      | NonNegative -> 
          Tensor.relu tensor
      | Sumto1 ->
          let sum = Tensor.sum tensor ~dim:0 in
          Tensor.div tensor sum
      | Range (min_val, max_val) ->
          let clipped = Tensor.clamp tensor 
            ~min:(Scalar.float min_val) 
            ~max:(Scalar.float max_val) in
          Tensor.div clipped (Tensor.sum clipped ~dim:0)
      | Custom f ->
          if f tensor then tensor
          else Tensor.div tensor (Tensor.sum tensor ~dim:0)
  end

  module EmissionModel = struct
    type emission_type =
      | Gaussian
      | StudentT of float
      | GaussianMixture of int

    let compute_emission state obs = function
      | Gaussian ->
          begin match state.model with
          | `GARCH params ->
              let pred = Garch.predict params 
                ~prev_volatility:obs.volatility 
                ~prev_return:obs.log_return in
              let diff = obs.volatility -. pred in
              Tensor.float (exp (-0.5 *. (diff *. diff)))
          | `FNN net ->
              let input = Tensor.of_float2 [|[|obs.volatility; obs.log_return|]|] in
              let pred = FNN.forward net input in
              let diff = Tensor.sub (Tensor.float obs.volatility) pred in
              Tensor.exp (Tensor.neg (Tensor.mul diff diff))
          end
      | StudentT df ->
          let gaussian = compute_emission state obs Gaussian in
          let df_tensor = Tensor.full_like gaussian df in
          Tensor.div gaussian (Tensor.sqrt df_tensor)
      | GaussianMixture n ->
          let components = List.init n (fun _ ->
            compute_emission state obs Gaussian
          ) in
          Tensor.mean (Tensor.stack components ~dim:0)
  end

  let create n_states =
    (* Initialize states with alternating GARCH and FNN models *)
    let states = Array.init n_states (fun i ->
      let model = if i mod 2 = 0 then
        `GARCH {omega = 0.1; alpha = 0.1; beta = 0.8}
      else
        `FNN (FNN.create_fnn2 ())
      in
      {id = i; model}
    ) in
    
    (* Initialize transition probabilities *)
    let transition_probs = 
      let t = Tensor.uniform [n_states; n_states] ~low:0. ~high:1. in
      Tensor.div t (Tensor.sum t ~dim:1 ~keepdim:true)
    in
    
    (* Initialize uniform initial probabilities *)
    let initial_probs = 
      Tensor.full [n_states] (1. /. float n_states)
    in
    
    {states; transition_probs; initial_probs}

  let compute_state_probs pmc obs =
    let open Torch in
    let n_states = Array.length pmc.states in
    let probs = Tensor.zeros [n_states] in
    
    (* Compute emission probabilities for each state *)
    Array.iteri (fun i state ->
      let emission = EmissionModel.compute_emission state obs Gaussian in
      let state_prob = Tensor.mul emission 
        (Tensor.select pmc.initial_probs ~dim:0 ~index:i) in
      let _ = Tensor.copy_ (Tensor.select probs ~dim:0 ~index:i) state_prob in
      ()
    ) pmc.states;
    
    (* Normalize probabilities *)
    let sum = Tensor.sum probs in
    Tensor.div probs sum

  let predict pmc observations =
    let open Torch in
    let n_states = Array.length pmc.states in
    let n_obs = List.length observations in
    let predictions = Tensor.zeros [n_obs] in
    
    (* Forward algorithm *)
    let alpha = Tensor.zeros [n_obs; n_states] in
    
    (* Initialize first step *)
    let first_obs = List.hd observations in
    Array.iteri (fun i state ->
      let emission = EmissionModel.compute_emission state first_obs Gaussian in
      let init_prob = Tensor.select pmc.initial_probs ~dim:0 ~index:i in
      let state_prob = Tensor.mul emission init_prob in
      let _ = Tensor.set_ alpha [0; i] (Tensor.to_float0_exn state_prob) in
      ()
    ) pmc.states;
    
    (* Forward pass *)
    List.iteri (fun t obs ->
      if t > 0 then
        for j = 0 to n_states - 1 do
          let sum_term = ref 0. in
          for i = 0 to n_states - 1 do
            let prev_alpha = Tensor.get alpha [t-1; i] |> Tensor.to_float0_exn in
            let trans_prob = Tensor.get pmc.transition_probs [i; j] 
              |> Tensor.to_float0_exn in
            sum_term := !sum_term +. prev_alpha *. trans_prob
          done;
          let emission = EmissionModel.compute_emission pmc.states.(j) obs Gaussian in
          let state_prob = Tensor.mul (Tensor.float !sum_term) emission in
          let _ = Tensor.set_ alpha [t; j] (Tensor.to_float0_exn state_prob) in
          ()
        done
    ) observations;
    
    (* Make predictions *)
    List.iteri (fun t obs ->
      let state_probs = Tensor.select alpha ~dim:0 ~index:t in
      let normalized_probs = Tensor.div state_probs (Tensor.sum state_probs) in
      
      (* Compute weighted prediction *)
      let pred = ref 0. in
      Array.iteri (fun i state ->
        let weight = Tensor.get normalized_probs [i] |> Tensor.to_float0_exn in
        let model_pred = match state.model with
          | `GARCH params ->
              Garch.predict params 
                ~prev_volatility:obs.volatility 
                ~prev_return:obs.log_return
          | `FNN net ->
              let input = Tensor.of_float2 [|[|obs.volatility; obs.log_return|]|] in
              FNN.forward net input |> Tensor.to_float0_exn
        in
        pred := !pred +. weight *. model_pred
      ) pmc.states;
      
      Tensor.set_ predictions [t] !pred
    ) observations;
    
    predictions

  let train pmc config data =
    let open Torch in
    
    let rec train_epoch state curr_pmc =
      if state.epoch >= config.max_epochs ||
         state.patience_counter >= config.early_stopping_patience then
        curr_pmc
      else
        (* E-step: compute state probabilities *)
        let n_obs = List.length data in
        let alpha = Tensor.zeros [n_obs; Array.length curr_pmc.states] in
        let beta = Tensor.zeros [n_obs; Array.length curr_pmc.states] in
        
        (* Forward pass *)
        List.iteri (fun t obs ->
          Array.iteri (fun i state ->
            let emission = EmissionModel.compute_emission state obs Gaussian in
            if t = 0 then
              let init_prob = Tensor.select curr_pmc.initial_probs ~dim:0 ~index:i in
              let _ = Tensor.set_ alpha [t; i] 
                (Tensor.to_float0_exn (Tensor.mul emission init_prob)) in
              ()
            else
              let sum_term = ref 0. in
              for j = 0 to Array.length curr_pmc.states - 1 do
                let prev_alpha = Tensor.get alpha [t-1; j] |> Tensor.to_float0_exn in
                let trans_prob = Tensor.get curr_pmc.transition_probs [j; i] 
                  |> Tensor.to_float0_exn in
                sum_term := !sum_term +. prev_alpha *. trans_prob
              done;
              let _ = Tensor.set_ alpha [t; i] 
                (Tensor.to_float0_exn (Tensor.mul emission (Tensor.float !sum_term))) in
              ()
          ) curr_pmc.states
        ) data;
        
        (* Backward pass *)
        for t = n_obs - 1 downto 0 do
          Array.iteri (fun i state ->
            if t = n_obs - 1 then
              let _ = Tensor.set_ beta [t; i] 1. in
              ()
            else
              let sum_term = ref 0. in
              for j = 0 to Array.length curr_pmc.states - 1 do
                let next_beta = Tensor.get beta [t+1; j] |> Tensor.to_float0_exn in
                let trans_prob = Tensor.get curr_pmc.transition_probs [i; j] 
                  |> Tensor.to_float0_exn in
                let emission = EmissionModel.compute_emission state 
                  (List.nth data (t+1)) Gaussian in
                sum_term := !sum_term +. next_beta *. trans_prob *. 
                  (Tensor.to_float0_exn emission)
              done;
              let _ = Tensor.set_ beta [t; i] !sum_term in
              ()
          ) curr_pmc.states
        done;
        
        (* M-step: update model parameters *)
        (* Update transition probabilities *)
        let new_trans = Tensor.zeros_like curr_pmc.transition_probs in
        for i = 0 to Array.length curr_pmc.states - 1 do
          for j = 0 to Array.length curr_pmc.states - 1 do
            let numer = ref 0. in
            let denom = ref 0. in
            List.iteri (fun t obs ->
              if t < n_obs - 1 then begin
                let alpha_i = Tensor.get alpha [t; i] |> Tensor.to_float0_exn in
                let beta_j = Tensor.get beta [t+1; j] |> Tensor.to_float0_exn in
                let trans_ij = Tensor.get curr_pmc.transition_probs [i; j] 
                  |> Tensor.to_float0_exn in
                let emission = EmissionModel.compute_emission curr_pmc.states.(j) 
                  (List.nth data (t+1)) Gaussian in
                numer := !numer +. alpha_i *. trans_ij *. beta_j *. 
                  (Tensor.to_float0_exn emission);
                denom := !denom +. alpha_i *. (Tensor.get beta [t; i] |> Tensor.to_float0_exn)
              end
            ) data;
            let _ = Tensor.set_ new_trans [i; j] (!numer /. max !denom 1e-10) in
            ()
          done
        done;
        
        (* Update models *)
        Array.iteri (fun i state ->
          match state.model with
          | `GARCH params ->
              let weighted_data = List.mapi (fun t obs ->
                let weight = (Tensor.get alpha [t; i] |> Tensor.to_float0_exn) *.
                           (Tensor.get beta [t; i] |> Tensor.to_float0_exn) in
                obs, weight
              ) data in
              let new_params = Garch.train (List.map fst weighted_data)
                config.learning_rate config.max_epochs in
              state.model <- `GARCH new_params
          | `FNN net ->
              let weighted_data = List.mapi (fun t obs ->
                let weight = (Tensor.get alpha [t; i] |> Tensor.to_float0_exn) *.
                           (Tensor.get beta [t; i] |> Tensor.to_float0_exn) in
                obs, weight
              ) data in
              let x, y = data_to_tensors (List.map fst weighted_data) in
              let weights = Tensor.of_float1 (Array.of_list 
                (List.map snd weighted_data)) in
              let _ = FNN.train net config (x, Tensor.mul y weights) in
              ()
        ) curr_pmc.states;
        
        (* Compute loss *)
        let predictions = predict curr_pmc data in
        let targets = Tensor.stack (List.map (fun obs ->
          Tensor.float obs.volatility) data) in
        let loss = match config.loss_fn with
          | MSE -> 
              let diff = Tensor.sub predictions targets in
              Tensor.mean (Tensor.mul diff diff)
          | MAE ->
              Tensor.mean (Tensor.abs (Tensor.sub predictions targets))
          | RMSE ->
              let diff = Tensor.sub predictions targets in
              Tensor.sqrt (Tensor.mean (Tensor.mul diff diff))
        in
        
        let loss_val = Tensor.to_float0_exn loss in
        let new_state =
          if loss_val < state.best_loss then
            {epoch = state.epoch + 1;
             loss = loss_val;
             best_loss = loss_val;
             patience_counter = 0}
          else
            {epoch = state.epoch + 1;
             loss = loss_val;
             best_loss = state.best_loss;
             patience_counter = state.patience_counter + 1}
        in
        
        train_epoch new_state {curr_pmc with transition_probs = new_trans}
    in
    
    train_epoch {
      epoch = 0;
      loss = Float.infinity;
      best_loss = Float.infinity;
      patience_counter = 0;
    } pmc
end

module PMCGarch = struct
  type model = {
    pmc: PMC.t;
    base_garch: Types.garch_params;
  }

  let create n_states =
    {
      pmc = PMC.create n_states;
      base_garch = {omega = 0.1; alpha = 0.1; beta = 0.8};
    }

  let predict model observations =
    let pmc_pred = PMC.predict model.pmc observations in
    let garch_pred = Tensor.stack (List.map (fun obs ->
      Tensor.float (Garch.predict model.base_garch
        ~prev_volatility:obs.volatility
        ~prev_return:obs.log_return)
    ) observations) in
    
    (* Combine predictions using state probabilities *)
    let weights = PMC.compute_state_probs model.pmc (List.hd observations) in
    let combined = Tensor.(add
      (mul pmc_pred (select weights ~dim:0 ~index:0))
      (mul garch_pred (select weights ~dim:0 ~index:1))) in
    combined

  let train model config data =
    let trained_pmc = PMC.train model.pmc config data in
    let trained_garch = Garch.train_with_constraints data 
      config.learning_rate config.max_epochs in
    {pmc = trained_pmc; base_garch = trained_garch}
end

module Metrics = struct
  type evaluation = {
    mape: float;
    rmse: float;
    dir_acc: float;
  }

  let mape pred target =
    let diff = Tensor.sub pred target in
    let abs_perc = Tensor.div (Tensor.abs diff) target in
    Tensor.mul (Tensor.mean abs_perc) (Scalar.float 100.0)

  let rmse pred target =
    let diff = Tensor.sub pred target in
    let squared = Tensor.mul diff diff in
    let mean = Tensor.mean squared in
    Tensor.sqrt mean

  let directional_accuracy pred target =
    let pred_diff = Tensor.diff pred ~dim:0 ~n:1 in
    let target_diff = Tensor.diff target ~dim:0 ~n:1 in
    let pred_direction = Tensor.sign pred_diff in
    let target_direction = Tensor.sign target_diff in
    let matches = Tensor.eq pred_direction target_direction in
    Tensor.to_float0_exn (Tensor.mean matches)

  let evaluate pred target =
    {
      mape = Tensor.to_float0_exn (mape pred target);
      rmse = Tensor.to_float0_exn (rmse pred target);
      dir_acc = directional_accuracy pred target;
    }
end

module Testing = struct
  type test_result = {
    model_name: string;
    metrics: Metrics.evaluation;
    predictions: Tensor.t;
    actual: Tensor.t;
  }

  let split_data data train_ratio valid_ratio =
    let total_size = List.length data in
    let train_size = int_of_float (float total_size *. train_ratio) in
    let valid_size = int_of_float (float total_size *. valid_ratio) in
    let test_size = total_size - train_size - valid_size in
    
    let train_data = List.take train_size data in
    let valid_data = List.sub data train_size valid_size in
    let test_data = List.sub data (train_size + valid_size) test_size in
    
    train_data, valid_data, test_data

  let cross_validate model data n_folds config =
    let total_size = List.length data in
    let fold_size = total_size / n_folds in
    
    let results = List.init n_folds (fun fold ->
      (* Create training and validation sets *)
      let val_start = fold * fold_size in
      let val_end = min (val_start + fold_size) total_size in
      
      let val_data = List.sub data val_start (val_end - val_start) in
      let train_data = List.concat [
        List.sub data 0 val_start;
        List.sub data val_end (total_size - val_end)
      ] in
      
      (* Train model *)
      let trained_model = match model with
        | `PMCGarch m -> PMCGarch.train m config train_data
        | `GARCH m -> Garch.train_with_constraints train_data 
            config.learning_rate config.max_epochs
        | `FNN m -> 
            let x, y = data_to_tensors train_data in
            let state = FNN.train m config (x, y) in
            m (* Return original model since FNN modifies in place *)
      in
      
      (* Evaluate on validation set *)
      let predictions = match trained_model with
        | `PMCGarch m -> PMCGarch.predict m val_data
        | `GARCH m -> Garch.predict_batch m (data_to_tensor val_data)
        | `FNN m -> 
            let x, _ = data_to_tensors val_data in
            FNN.forward m x
      in
      
      let actual = Tensor.stack 
        (List.map (fun obs -> Tensor.float obs.volatility) val_data) in
      
      Metrics.evaluate predictions actual
    ) in
    
    results

  let compare_models models data config =
    let train_data, valid_data, test_data = split_data data 0.6 0.2 in
    
    List.map (fun (name, model) ->
      (* Train model *)
      let trained_model = match model with
        | `PMCGarch m -> 
            let trained = PMCGarch.train m config train_data in
            `PMCGarch trained
        | `GARCH m ->
            let trained = Garch.train_with_constraints train_data 
              config.learning_rate config.max_epochs in
            `GARCH trained
        | `FNN m ->
            let x, y = data_to_tensors train_data in
            let _ = FNN.train m config (x, y) in
            `FNN m
      in
      
      (* Get predictions on test set *)
      let predictions = match trained_model with
        | `PMCGarch m -> PMCGarch.predict m test_data
        | `GARCH m -> Garch.predict_batch m (data_to_tensor test_data)
        | `FNN m -> 
            let x, _ = data_to_tensors test_data in
            FNN.forward m x
      in
      
      let actual = Tensor.stack 
        (List.map (fun obs -> Tensor.float obs.volatility) test_data) in
      
      let metrics = Metrics.evaluate predictions actual in
      
      {
        model_name = name;
        metrics;
        predictions;
        actual;
      }
    ) models

  let detect_regimes volatility =
    let mean_vol = Tensor.mean volatility in
    let std_vol = Tensor.std volatility in
    let high_threshold = Tensor.add mean_vol 
      (Tensor.mul std_vol (Scalar.float 1.5)) in
    let low_threshold = Tensor.sub mean_vol 
      (Tensor.mul std_vol (Scalar.float 1.5)) in
    
    let regimes = Tensor.zeros_like volatility in
    let _ = Tensor.where_ regimes
      ~condition:(Tensor.gt volatility high_threshold)
      ~other:(Scalar.int 2)  (* High volatility regime *)
      ~self:(Scalar.int 1) in (* Normal regime *)
    let _ = Tensor.where_ regimes
      ~condition:(Tensor.lt volatility low_threshold)
      ~other:(Scalar.int 0)  (* Low volatility regime *)
      ~self:(Scalar.int 1) in
    regimes

  module StatTests = struct
    let diebold_mariano pred1 pred2 actual =
      let diff1 = Tensor.sub pred1 actual in
      let diff2 = Tensor.sub pred2 actual in
      let d = Tensor.sub (Tensor.mul diff1 diff1) (Tensor.mul diff2 diff2) in
      let mean_d = Tensor.mean d in
      let var_d = Tensor.var d in
      let n = float (Tensor.size d 0) in
      Tensor.to_float0_exn (Tensor.div mean_d 
        (Tensor.sqrt (Tensor.div var_d (Scalar.float n))))

    let model_confidence_set models predictions actual confidence_level =
      let n_models = List.length models in
      let losses = List.map (fun pred ->
        let diff = Tensor.sub pred actual in
        Tensor.mul diff diff
      ) predictions in
      
      (* Compute test statistics *)
      let t_stats = List.map (fun loss ->
        let mean_loss = Tensor.mean loss in
        let std_loss = Tensor.std loss in
        let n = float (Tensor.size loss 0) in
        Tensor.to_float0_exn (Tensor.div mean_loss 
          (Tensor.div std_loss (Scalar.float (sqrt n))))
      ) losses in
      
      (* Select models based on confidence level *)
      List.filter2 (fun model t_stat ->
        abs_float t_stat <= Stats.normal_ppf confidence_level
      ) models t_stats
  end

  let default_config = {
    max_epochs = 100;
    batch_size = 32;
    learning_rate = 0.001;
    momentum = 0.9;
    early_stopping_patience = 10;
    loss_fn = RMSE;
  }

  let create_model_suite n_states features =
    [
      "GARCH(1,1)", `GARCH {omega = 0.1; alpha = 0.1; beta = 0.8};
      "FNN(2)", `FNN (FNN.create_fnn2 ());
      "FNN(3)", `FNN (FNN.create_fnn3 ());
      "FNN(2,3)", `FNN (FNN.create_fnn2_3 ());
      "PMC-GARCH", `PMCGarch (PMCGarch.create n_states);
    ]

  let run_volatility_experiment ?(n_states=3) ?(window_size=20) data_file =
    (* Load and preprocess data *)
    let market_data = FinancialData.load_market_data data_file in
    let features = FinancialData.create_features market_data window_size in
    
    (* Create models *)
    let models = create_model_suite n_states features in
    
    (* Run comparison *)
    let results = Testing.compare_models models 
      (create_windows [] window_size) default_config in
    
    (* Cross validation *)
    let cv_results = List.map (fun (name, model) ->
      name, Testing.cross_validate model 
        (create_windows [] window_size) 5 default_config
    ) models in
    
    results, cv_results

  let run_comprehensive_experiment ?(n_states=3) ?(window_size=20) data_file =
    let market_data = FinancialData.load_market_data data_file in
    let features = FinancialData.create_features market_data window_size in
    let models = create_model_suite n_states features in
    
    (* Basic experiments *)
    let basic_results, cv_results = 
      run_volatility_experiment ~n_states ~window_size data_file in
    
    (* Additional analyses *)
    let regime_analysis = Results.detect_regimes 
      (FinancialData.calc_realized_volatility market_data.close_prices 5) in
    
    let dm_tests = List.map2 (fun r1 r2 ->
      Results.StatTests.diebold_mariano r1.predictions r2.predictions r1.actual
    ) basic_results (List.tl basic_results) in
    
    let mcs_models = Results.StatTests.model_confidence_set
      models
      (List.map (fun r -> r.predictions) basic_results)
      (List.hd basic_results).actual
      0.95 in
    
    basic_results, cv_results, regime_analysis, dm_tests, mcs_models
end