open Torch

type observation_panel = {
  g : int;    (* Group indicator (0 or 1) *)
  d : int;    (* Domain indicator (0 or 1) *)
  x : Tensor.t; (* Covariates *)
  y0 : float; (* Outcome at time 0 *)
  y1 : float; (* Outcome at time 1 *)
}

type observation_rc = {
  g : int;    (* Group indicator (0 or 1) *)
  d : int;    (* Domain indicator (0 or 1) *)
  t : int;    (* Time indicator (0 or 1) *)
  x : Tensor.t; (* Covariates *)
  y : float;  (* Outcome *)
}

(* Estimator panel signature *)
module type ESTIMATOR_PANEL = sig
  type t

  val create : ?config:string -> unit -> t
  val fit : t -> observation_panel array -> t
  val predict : t -> Tensor.t -> float
end

(* Neural Network-based outcome regression estimators *)
module NeuralOutcomeEstimator = struct
  (* Panel data outcome regression model *)
  module Panel = struct
    type model = {
      net: Module.t;
      optimizer: Optimizer.t;
      learning_rate: float;
      batch_size: int;
      epochs: int;
    }
    
    (* Create a neural network for estimating E[Y1 - Y0 | X, G, D] *)
    let create ?(learning_rate=0.001) ?(batch_size=32) ?(epochs=100) ?(hidden_dim=64) input_dim =
      let vs = Var_store.create ~name:"outcome_nn" () in
      
      let net = Layer.sequential vs [
        Layer.linear vs ~input_dim ~output_dim:hidden_dim;
        Layer.relu;
        Layer.linear vs ~input_dim:hidden_dim ~output_dim:1;
      ] in
      
      let optimizer = Optimizer.adam vs ~learning_rate in
      
      { net; optimizer; learning_rate; batch_size; epochs }
    
    (* Train the model on panel data observations *)
    let train model obs_array g d =
      (* Filter observations for the specified group (g, d) *)
      let filtered_obs = Utils.filter_by_gd obs_array g d in
      let n = Array.length filtered_obs in
      
      if n = 0 then model else begin
        (* Extract X and Y1-Y0 *)
        let x_data = Array.map (fun o -> o.x) filtered_obs in
        let y_data = Array.map (fun o -> o.y1 -. o.y0) filtered_obs in
        
        (* Convert to tensors *)
        let x_tensor = Tensor.stack (Array.to_list x_data) ~dim:0 in
        let y_tensor = Tensor.of_float1 y_data |> Tensor.reshape ~shape:[-1; 1] in
        
        (* Training loop *)
        for epoch = 1 to model.epochs do
          (* Initialize loss for this epoch *)
          let epoch_loss = ref 0.0 in
          
          (* Process data in batches *)
          for i = 0 to n / model.batch_size - 1 do
            let start_idx = i * model.batch_size in
            let end_idx = min n (start_idx + model.batch_size) in
            let batch_size = end_idx - start_idx in
            
            (* Extract batch data *)
            let x_batch = Tensor.narrow x_tensor ~dim:0 ~start:start_idx ~length:batch_size in
            let y_batch = Tensor.narrow y_tensor ~dim:0 ~start:start_idx ~length:batch_size in
            
            (* Forward pass *)
            let y_pred = Module.forward model.net x_batch in
            
            (* Compute MSE loss *)
            let loss = Tensor.mse_loss y_pred y_batch in
            
            (* Backward pass and optimize *)
            Optimizer.zero_grad model.optimizer;
            Tensor.backward loss;
            Optimizer.step model.optimizer;
            
            (* Accumulate loss *)
            epoch_loss := !epoch_loss +. Tensor.to_float0_exn loss;
          done;
          
          (* Print progress every 10 epochs *)
          if epoch mod 10 = 0 then
            Printf.printf "Epoch %d/%d, Loss: %.6f\n" 
              epoch model.epochs (!epoch_loss /. float_of_int (n / model.batch_size));
        done;
        
        model
      end
    
    (* Predict on new data *)
    let predict model x =
      let y_pred = Module.forward model.net x in
      Tensor.to_float0_exn y_pred
    
    (* Get a function that predicts μg,d,Δ(X) *)
    let get_mu_estimator model_map =
      fun g d x -> 
        try 
          let model = Hashtbl.find model_map (g, d) in
          predict model x
        with Not_found -> 0.0  (* Default if model not found *)
  end

  (* Repeated cross-sections outcome regression model *)
  module RC = struct
    type model = {
      net: Module.t;
      optimizer: Optimizer.t;
      learning_rate: float;
      batch_size: int;
      epochs: int;
    }
    
    (* Create a neural network for estimating E[Y | X, G, D, T] *)
    let create ?(learning_rate=0.001) ?(batch_size=32) ?(epochs=100) ?(hidden_dim=64) input_dim =
      let vs = Var_store.create ~name:"outcome_rc_nn" () in
      
      let net = Layer.sequential vs [
        Layer.linear vs ~input_dim ~output_dim:hidden_dim;
        Layer.relu;
        Layer.linear vs ~input_dim:hidden_dim ~output_dim:1;
      ] in
      
      let optimizer = Optimizer.adam vs ~learning_rate in
      
      { net; optimizer; learning_rate; batch_size; epochs }
    
    (* Train the model on repeated cross-sections data *)
    let train model obs_array g d t =
      (* Filter observations for the specified group (g, d, t) *)
      let filtered_obs = Utils.filter_observations obs_array 
        (fun o -> o.g = g && o.d = d && o.t = t) in
      let n = Array.length filtered_obs in
      
      if n = 0 then model else begin
        (* Extract X and Y *)
        let x_data = Array.map (fun o -> o.x) filtered_obs in
        let y_data = Array.map (fun o -> o.y) filtered_obs in
        
        (* Convert to tensors *)
        let x_tensor = Tensor.stack (Array.to_list x_data) ~dim:0 in
        let y_tensor = Tensor.of_float1 y_data |> Tensor.reshape ~shape:[-1; 1] in
        
        (* Training loop *)
        for epoch = 1 to model.epochs do
          (* Initialize loss for this epoch *)
          let epoch_loss = ref 0.0 in
          
          (* Process data in batches *)
          for i = 0 to n / model.batch_size - 1 do
            let start_idx = i * model.batch_size in
            let end_idx = min n (start_idx + model.batch_size) in
            let batch_size = end_idx - start_idx in
            
            (* Extract batch data *)
            let x_batch = Tensor.narrow x_tensor ~dim:0 ~start:start_idx ~length:batch_size in
            let y_batch = Tensor.narrow y_tensor ~dim:0 ~start:start_idx ~length:batch_size in
            
            (* Forward pass *)
            let y_pred = Module.forward model.net x_batch in
            
            (* Compute MSE loss *)
            let loss = Tensor.mse_loss y_pred y_batch in
            
            (* Backward pass and optimize *)
            Optimizer.zero_grad model.optimizer;
            Tensor.backward loss;
            Optimizer.step model.optimizer;
            
            (* Accumulate loss *)
            epoch_loss := !epoch_loss +. Tensor.to_float0_exn loss;
          done;
          
          (* Print progress every 10 epochs *)
          if epoch mod 10 = 0 then
            Printf.printf "Epoch %d/%d, Loss: %.6f\n" 
              epoch model.epochs (!epoch_loss /. float_of_int (n / model.batch_size));
        done;
        
        model
      end
    
    (* Predict on new data *)
    let predict model x =
      let y_pred = Module.forward model.net x in
      Tensor.to_float0_exn y_pred
    
    (* Get a function that predicts μg,d,t(X) *)
    let get_mu_estimator model_map =
      fun g d t x -> 
        try 
          let model = Hashtbl.find model_map (g, d, t) in
          predict model x
        with Not_found -> 0.0  (* Default if model not found *)
  end
end

(* Neural Network-based propensity score estimators *)
module NeuralPropensityEstimator = struct
  (* Panel data propensity score model *)
  module Panel = struct
    type model = {
      net: Module.t;
      optimizer: Optimizer.t;
      learning_rate: float;
      batch_size: int;
      epochs: int;
    }
    
    (* Create a neural network for estimating p(G=g, D=d | X) *)
    let create ?(learning_rate=0.001) ?(batch_size=32) ?(epochs=100) ?(hidden_dim=64) input_dim =
      let vs = Var_store.create ~name:"propensity_nn" () in
      
      let net = Layer.sequential vs [
        Layer.linear vs ~input_dim ~output_dim:hidden_dim;
        Layer.relu;
        Layer.linear vs ~input_dim:hidden_dim ~output_dim:1;
        Layer.sigmoid;
      ] in
      
      let optimizer = Optimizer.adam vs ~learning_rate in
      
      { net; optimizer; learning_rate; batch_size; epochs }
    
    (* Train the model to estimate p(G=g, D=d | X) *)
    let train model obs_array g d =
      let n = Array.length obs_array in
      
      (* Extract X and binary indicators for G=g, D=d *)
      let x_data = Array.map (fun o -> o.x) obs_array in
      let y_data = Array.map (fun o -> 
          if o.g = g && o.d = d then 1.0 else 0.0) obs_array in
      
      (* Convert to tensors *)
      let x_tensor = Tensor.stack (Array.to_list x_data) ~dim:0 in
      let y_tensor = Tensor.of_float1 y_data |> Tensor.reshape ~shape:[-1; 1] in
      
      (* Training loop *)
      for epoch = 1 to model.epochs do
        (* Initialize loss for this epoch *)
        let epoch_loss = ref 0.0 in
        
        (* Process data in batches *)
        for i = 0 to n / model.batch_size - 1 do
          let start_idx = i * model.batch_size in
          let end_idx = min n (start_idx + model.batch_size) in
          let batch_size = end_idx - start_idx in
          
          (* Extract batch data *)
          let x_batch = Tensor.narrow x_tensor ~dim:0 ~start:start_idx ~length:batch_size in
          let y_batch = Tensor.narrow y_tensor ~dim:0 ~start:start_idx ~length:batch_size in
          
          (* Forward pass *)
          let y_pred = Module.forward model.net x_batch in
          
          (* Compute binary cross-entropy loss *)
          let loss = Tensor.binary_cross_entropy y_pred y_batch in
          
          (* Backward pass and optimize *)
          Optimizer.zero_grad model.optimizer;
          Tensor.backward loss;
          Optimizer.step model.optimizer;
          
          (* Accumulate loss *)
          epoch_loss := !epoch_loss +. Tensor.to_float0_exn loss;
        done;
        
        (* Print progress every 10 epochs *)
        if epoch mod 10 = 0 then
          Printf.printf "Epoch %d/%d, Loss: %.6f\n" 
            epoch model.epochs (!epoch_loss /. float_of_int (n / model.batch_size));
      done;
      
      model
    
    (* Predict propensity score p(G=g, D=d | X) *)
    let predict model x =
      let pred = Module.forward model.net x in
      let p = Tensor.to_float0_exn pred in
      (* Ensure probability is bounded away from 0 and 1 for numerical stability *)
      max 0.001 (min 0.999 p)
    
    (* Get a function that predicts π_g,d(X) *)
    let get_pi_estimator model_map =
      fun g d x -> 
        try 
          let model = Hashtbl.find model_map (g, d) in
          predict model x
        with Not_found -> 0.25  (* Default if model not found = 1/4 probability *)
  end

  (* Repeated cross-sections propensity score model *)
  module RC = struct
    type model = {
      net: Module.t;
      optimizer: Optimizer.t;
      learning_rate: float;
      batch_size: int;
      epochs: int;
    }
    
    (* Create a neural network for estimating p(G=g, D=d, T=t | X) *)
    let create ?(learning_rate=0.001) ?(batch_size=32) ?(epochs=100) ?(hidden_dim=64) input_dim =
      let vs = Var_store.create ~name:"propensity_rc_nn" () in
      
      let net = Layer.sequential vs [
        Layer.linear vs ~input_dim ~output_dim:hidden_dim;
        Layer.relu;
        Layer.linear vs ~input_dim:hidden_dim ~output_dim:1;
        Layer.sigmoid;
      ] in
      
      let optimizer = Optimizer.adam vs ~learning_rate in
      
      { net; optimizer; learning_rate; batch_size; epochs }
    
    (* Train the model to estimate p(G=g, D=d, T=t | X) *)
    let train model obs_array g d t =
      let n = Array.length obs_array in
      
      (* Extract X and binary indicators for G=g, D=d, T=t *)
      let x_data = Array.map (fun o -> o.x) obs_array in
      let y_data = Array.map (fun o -> 
          if o.g = g && o.d = d && o.t = t then 1.0 else 0.0) obs_array in
      
      (* Convert to tensors *)
      let x_tensor = Tensor.stack (Array.to_list x_data) ~dim:0 in
      let y_tensor = Tensor.of_float1 y_data |> Tensor.reshape ~shape:[-1; 1] in
      
      (* Training loop *)
      for epoch = 1 to model.epochs do
        (* Initialize loss for this epoch *)
        let epoch_loss = ref 0.0 in
        
        (* Process data in batches *)
        for i = 0 to n / model.batch_size - 1 do
          let start_idx = i * model.batch_size in
          let end_idx = min n (start_idx + model.batch_size) in
          let batch_size = end_idx - start_idx in
          
          (* Extract batch data *)
          let x_batch = Tensor.narrow x_tensor ~dim:0 ~start:start_idx ~length:batch_size in
          let y_batch = Tensor.narrow y_tensor ~dim:0 ~start:start_idx ~length:batch_size in
          
          (* Forward pass *)
          let y_pred = Module.forward model.net x_batch in
          
          (* Compute binary cross-entropy loss *)
          let loss = Tensor.binary_cross_entropy y_pred y_batch in
          
          (* Backward pass and optimize *)
          Optimizer.zero_grad model.optimizer;
          Tensor.backward loss;
          Optimizer.step model.optimizer;
          
          (* Accumulate loss *)
          epoch_loss := !epoch_loss +. Tensor.to_float0_exn loss;
        done;
        
        (* Print progress every 10 epochs *)
        if epoch mod 10 = 0 then
          Printf.printf "Epoch %d/%d, Loss: %.6f\n" 
            epoch model.epochs (!epoch_loss /. float_of_int (n / model.batch_size));
      done;
      
      model
    
    (* Predict propensity score p(G=g, D=d, T=t | X) *)
    let predict model x =
      let pred = Module.forward model.net x in
      let p = Tensor.to_float0_exn pred in
      (* Ensure probability is bounded away from 0 and 1 for numerical stability *)
      max 0.001 (min 0.999 p)
    
    (* Get a function that predicts π_g,d,t(X) *)
    let get_pi_estimator model_map =
      fun g d t x -> 
        try 
          let model = Hashtbl.find model_map (g, d, t) in
          predict model x
        with Not_found -> 0.125  (* Default if model not found = 1/8 probability *)
  end

  (* No compositional changes assumption version *)
  module RC_NoCompChanges = struct
    (* Reuse the Panel module since under no compositional changes,
       p(G=g, D=d | X) is time-invariant *)
    
    (* Estimate p(G=g, D=d | X) using data from both time periods *)
    let train_pooled model obs_array g d =
      let n = Array.length obs_array in
      
      (* Extract X and binary indicators for G=g, D=d, pooling over T *)
      let x_data = Array.map (fun o -> o.x) obs_array in
      let y_data = Array.map (fun o -> 
          if o.g = g && o.d = d then 1.0 else 0.0) obs_array in
      
      (* Convert to tensors *)
      let x_tensor = Tensor.stack (Array.to_list x_data) ~dim:0 in
      let y_tensor = Tensor.of_float1 y_data |> Tensor.reshape ~shape:[-1; 1] in
      
      for epoch = 1 to model.epochs do
        (* Initialize loss for this epoch *)
        let epoch_loss = ref 0.0 in
        
        (* Process data in batches *)
        for i = 0 to n / model.batch_size - 1 do
          let start_idx = i * model.batch_size in
          let end_idx = min n (start_idx + model.batch_size) in
          let batch_size = end_idx - start_idx in
          
          (* Extract batch data *)
          let x_batch = Tensor.narrow x_tensor ~dim:0 ~start:start_idx ~length:batch_size in
          let y_batch = Tensor.narrow y_tensor ~dim:0 ~start:start_idx ~length:batch_size in
          
          (* Forward pass *)
          let y_pred = Module.forward model.net x_batch in
          
          (* Compute binary cross-entropy loss *)
          let loss = Tensor.binary_cross_entropy y_pred y_batch in
          
          (* Backward pass and optimize *)
          Optimizer.zero_grad model.optimizer;
          Tensor.backward loss;
          Optimizer.step model.optimizer;
          
          (* Accumulate loss *)
          epoch_loss := !epoch_loss +. Tensor.to_float0_exn loss;
        done;
        
        (* Print progress every 10 epochs *)
        if epoch mod 10 = 0 then
          Printf.printf "Epoch %d/%d, Loss: %.6f\n" 
            epoch model.epochs (!epoch_loss /. float_of_int (n / model.batch_size));
      done;
      
      model
  end
end

(* Influence function for the outcome regression (OR) based ATT estimator *)
let if_att_or obs mu_estimator =

  (* Extract data *)
  let g = obs.g in
  let d = obs.d in
  let x = obs.x in
  let delta_y = obs.y1 -. obs.y0 in
  
  if g = 1 && d = 1 then
    (* For treated units (G=1, D=1) *)
    let mu_correction = 
      mu_estimator 0 1 x +. mu_estimator 1 0 x -. mu_estimator 0 0 x in
    delta_y -. mu_correction
  else
    (* For other units with different (g,d) combinations *)
    let sign = if (g + d + 1) mod 2 = 0 then 1.0 else -1.0 in
    sign *. (delta_y -. mu_estimator g d x)

(* Influence function for the inverse propensity weighting (IPW) based ATT estimator *)
let if_att_ipw obs pi_estimator =
  let open Utils in
  
  (* Extract data *)
  let g = obs.g in
  let d = obs.d in
  let x = obs.x in
  let delta_y = obs.y1 -. obs.y0 in
  
  (* Define ρ0(X, G, D) function from Equation (4) *)
  let rho0 =
    [|(0,0); (0,1); (1,0); (1,1)|]
    |> Array.map (fun (g',d') ->
        let numerator = (1 - g' - g) * (1 - d' - d) |> float_of_int in
        let denominator = pi_estimator g' d' x in
        numerator /. denominator)
    |> Array.fold_left (+.) 0.0
  in
  
  (* Calculate the influence function *)
  if g = 1 && d = 1 then
    (* For treated units (G=1, D=1) *)
    delta_y
  else
    (* For other units with different (g,d) combinations *)
    (float_of_int (g * d) -. pi_estimator 1 1 x *. rho0) *. delta_y

(* Influence function for the doubly robust (DR) ATT estimator *)
let if_att_dr obs mu_estimator pi_estimator =
  let open Utils in
  
  (* Extract data *)
  let g = obs.g in
  let d = obs.d in
  let x = obs.x in
  let delta_y = obs.y1 -. obs.y0 in
  
  (* Define the weight function w_g,d *)
  let w_gd g' d' =
    let indicator = if g = g' && d = d' then 1.0 else 0.0 in
    let pi_11_x = pi_estimator 1 1 x in
    let pi_gd_x = pi_estimator g' d' x in
    
    (float_of_int (g * d) -. pi_11_x *. indicator) /. pi_gd_x
  in
  
  (* Calculate the influence function *)
  [|(0,0); (0,1); (1,0); (1,1)|]
  |> Array.map (fun (g',d') ->
      let sign = if (g' + d' + 1) mod 2 = 0 then 1.0 else -1.0 in
      let weight = w_gd g' d' in
      sign *. weight *. (delta_y -. mu_estimator g' d' x))
  |> Array.fold_left (+.) 0.0
end

(* Influence function for the outcome regression (OR) based ATT estimator *)
let if_att_or obs mu_estimator =
  let open Utils in
  
  (* Extract data *)
  let g = obs.g in
  let d = obs.d in
  let t = obs.t in
  let x = obs.x in
  let y = obs.y in
  
  if g = 1 && d = 1 && t = 1 then
    (* For treated post-treatment units (G=1, D=1, T=1) *)
    let mu_correction = 
      mu_estimator 1 1 0 x +. 
      (mu_estimator 0 1 1 x -. mu_estimator 0 1 0 x) +. 
      (mu_estimator 1 0 1 x -. mu_estimator 1 0 0 x) -. 
      (mu_estimator 0 0 1 x -. mu_estimator 0 0 0 x) in
    y -. mu_correction
  else
    (* For other units with different (g,d,t) combinations *)
    let sign = if (g + d + t) mod 2 = 0 then 1.0 else -1.0 in
    sign *. (y -. mu_estimator g d t x)

module type ESTIMATOR_RC = sig
  type t

  val create : ?config:string -> unit -> t
  val fit : t -> observation_rc array -> t
  val predict : t -> Tensor.t -> float
end

(* Module signature for propensity score estimators *)
module type PROPENSITY_ESTIMATOR = sig
  type t

  val create : ?config:string -> unit -> t
  val fit : t -> Tensor.t array -> int array -> t
  val predict : t -> Tensor.t -> float
end

let mean_float arr =
    let sum = Array.fold_left (fun acc x -> acc +. x) 0. arr in
    sum /. (float_of_int (Array.length arr))

let filter_observations obs_panel pred =
    Array.to_seq obs_panel
    |> Seq.filter pred
    |> Array.of_seq

let filter_by_gd obs_panel g d =
    filter_observations obs_panel (fun o -> o.g = g && o.d = d)

let float_array_of_tensor t =
    let size = Tensor.shape t |> List.hd in
    Array.init size (fun i -> Tensor.get t [i] |> Float.of_int)

let mean_tensor t =
    Tensor.mean t ~dim:[0] ~keepdim:false ~dtype:(T Float)

(* Compute empirical mean *)
let empirical_mean data =
    let n = Array.length data in
    if n = 0 then 0. else mean_float data

(* Compute difference in outcome (Y1 - Y0) for panel data *)
let compute_delta_y obs_panel =
    Array.map (fun o -> o.y1 -. o.y0) obs_panel

(* Extract tensors from observations *)
let extract_covariates obs_panel =
    Array.map (fun o -> o.x) obs_panel

let extract_outcomes_panel obs_panel =
    Array.map (fun o -> (o.y0, o.y1)) obs_panel

let extract_outcomes_rc obs_panel =
    Array.map (fun o -> o.y) obs_panel

let extract_gd obs_panel =
    Array.map (fun o -> (o.g, o.d)) obs_panel

let extract_gdt obs_rc =
    Array.map (fun o -> (o.g, o.d, o.t)) obs_rc

(* Simple neural network-based outcome estimator *)
module NNOutcomeEstimator : ESTIMATOR_PANEL = struct
  type t = {
    model : Module.t;
    config : string;
  }

  let create ?(config="default") () =
    let vs = Var_store.create ~name:"outcome_estimator" () in
    let input_dim = 10 in (* 10 covariates *)
    let hidden_dim = 50 in
    let model =
      let open Layer in
      sequential vs [
        linear vs ~input_dim ~output_dim:hidden_dim;
        relu;
        linear vs ~input_dim:hidden_dim ~output_dim:1;
      ]
    in
    { model; config }

  let predict estimator x =
    (* Forward pass through the model *)
    let pred = Module.forward estimator.model x in
    Tensor.to_float0_exn pred
end

(* Simple neural network-based propensity score estimator *)
module NNPropensityEstimator : PROPENSITY_ESTIMATOR = struct
  type t = {
    model : Module.t;
    config : string;
  }

  let create ?(config="default") () =
    let vs = Var_store.create ~name:"propensity_estimator" () in
    let input_dim = 10 in
    let hidden_dim = 50 in
    let model =
      let open Layer in
      sequential vs [
        linear vs ~input_dim ~output_dim:hidden_dim;
        relu;
        linear vs ~input_dim:hidden_dim ~output_dim:1;
        sigmoid;
      ]
    in
    { model; config }

  let fit estimator x_array y_array =
    estimator

  let predict estimator x =
    let pred = Module.forward estimator.model x in
    Tensor.to_float0_exn pred
end

(* Compute the ATT for panel data *)
let compute_tau_p obs_panel =
  let treated = Utils.filter_by_gd obs_panel 1 1 in
  let delta_y = Utils.compute_delta_y treated in
  Utils.empirical_mean delta_y

(* Generate synthetic panel data for testing *)
let generate_synthetic_data ?(n=1000) ?(covariates_dim=10) () =
  let open Utils in
  let rng = Random.State.make_self_init () in
  Array.init n (fun _ ->
    let g = if Random.State.float rng 1.0 < 0.5 then 0 else 1 in
    let d = if Random.State.float rng 1.0 < 0.5 then 0 else 1 in
    let x = Tensor.rand [covariates_dim] ~device:Device.Cpu in
    let y0 = Random.State.float rng 10.0 in
    (* Treatment effect is 2.0 for g=1, d=1 *)
    let treatment_effect = if g = 1 && d = 1 then 2.0 else 0.0 in
    let y1 = y0 +. treatment_effect +. Random.State.float rng 1.0 in
    { g; d; x; y0; y1 }
  )

(* Compute the ATT for repeated cross-sections *)
let compute_tau_rc obs_rc =
  let treated_post = Utils.filter_observations obs_rc (fun o -> o.g = 1 && o.d = 1 && o.t = 1) in
  let treated_pre = Utils.filter_observations obs_rc (fun o -> o.g = 1 && o.d = 1 && o.t = 0) in
  let y_treated_post = Utils.extract_outcomes_rc treated_post in
  let y_treated_pre = Utils.extract_outcomes_rc treated_pre in
  Utils.empirical_mean y_treated_post -. Utils.empirical_mean y_treated_pre

(* Generate synthetic repeated cross-sections data for testing *)
let generate_synthetic_data ?(n=1000) ?(covariates_dim=10) () =
  let rng = Random.State.make_self_init () in
  Array.init n (fun _ ->
    let g = if Random.State.float rng 1.0 < 0.5 then 0 else 1 in
    let d = if Random.State.float rng 1.0 < 0.5 then 0 else 1 in
    let t = if Random.State.float rng 1.0 < 0.5 then 0 else 1 in
    let x = Tensor.rand [covariates_dim] ~device:Device.Cpu in
    (* Base outcome *)
    let base = Random.State.float rng 10.0 in
    (* Treatment effect is 2.0 for g=1, d=1, t=1 *)
    let treatment_effect = if g = 1 && d = 1 && t = 1 then 2.0 else 0.0 in
    let y = base +. treatment_effect +. Random.State.float rng 1.0 in
    { g; d; t; x; y }
  )

(* Outcome Regression (OR) based ATT identification *)
let identify_att_or obs_panel mu_estimator =      
  (* Calculate the empirical mean of Y1 - Y0 for the treated group *)
  let treated = filter_by_gd obs_panel 1 1 in
  let e_delta_y_treated = empirical_mean (compute_delta_y treated) in
  
  (* Estimate μ values for each group *)
  let estimate_mu g d x =
    let estimator = mu_estimator (g, d) in
    NNOutcomeEstimator.predict estimator x
  in
  
  (* Calculate E[μ0,1,Δ(X) + μ1,0,Δ(X) - μ0,0,Δ(X) | G = 1, D = 1] *)
  let e_mu_correction =
    treated
    |> Array.map (fun obs ->
        let mu_01 = estimate_mu 0 1 obs.x in
        let mu_10 = estimate_mu 1 0 obs.x in
        let mu_00 = estimate_mu 0 0 obs.x in
        mu_01 +. mu_10 -. mu_00)
    |> empirical_mean
  in
  
  (* ATT = E[Y1 - Y0 | G = 1, D = 1] - E[μ0,1,Δ(X) + μ1,0,Δ(X) - μ0,0,Δ(X) | G = 1, D = 1] *)
  e_delta_y_treated -. e_mu_correction

(* Inverse Propensity Weighting (IPW) based ATT identification *)
let identify_att_ipw obs_panel pi_estimator =  
  (* Calculate E[G·D] *)
  let gd_array = Array.map (fun o -> float_of_int (o.g * o.d)) obs_panel in
  let e_gd = empirical_mean gd_array in
  
  (* Define ρ0(X, G, D) function *)
  let rho0 obs pi_fn =
    let x = obs.x in
    let g = obs.g in
    let d = obs.d in
    
    (* Sum over all (g,d) combinations *)
    let sum_term = 
      [|(0,0); (0,1); (1,0); (1,1)|]
      |> Array.map (fun (g',d') ->
          let numerator = (1 - g' - g) * (1 - d' - d) |> float_of_int in
          let denominator = pi_fn g' d' x in
          numerator /. denominator)
      |> Array.fold_left (+.) 0.0
    in
    sum_term
  in
  
  (* Calculate the IPW estimator *)
  let weighted_sum =
    obs_panel
    |> Array.map (fun obs ->
        let delta_y = obs.y1 -. obs.y0 in
        let pi_11_x = pi_estimator 1 1 obs.x in
        let rho0_val = rho0 obs (fun g d x -> pi_estimator g d x) in
        (pi_11_x /. e_gd) *. rho0_val *. delta_y)
    |> Array.fold_left (+.) 0.0
  in
  
  weighted_sum /. (float_of_int (Array.length obs_panel))

(* Identification in the Repeated Cross-Sections Setting *)
module RepeatedCrossSections = struct
    (* Outcome Regression (OR) based ATT identification *)
    let identify_att_or obs_rc mu_estimator =      
      (* Get treated post-treatment group *)
      let treated_post = 
        filter_observations obs_rc (fun o -> o.g = 1 && o.d = 1 && o.t = 1) in
      
      (* Calculate E[Y | G = 1, D = 1, T = 1] *)
      let e_y_treated_post = 
        empirical_mean (extract_outcomes_rc treated_post) in
      
      (* Estimate conditional means μg,d,t(X) for different groups *)
      let estimate_mu g d t x =
        let estimator = mu_estimator (g, d, t) in
        NNOutcomeEstimator.predict estimator x
      in
      
      (* Calculate the correction term *)
      let correction_term =
        treated_post
        |> Array.map (fun obs ->
            let mu_110 = estimate_mu 1 1 0 obs.x in
            let mu_01delta = 
              estimate_mu 0 1 1 obs.x -. estimate_mu 0 1 0 obs.x in
            let mu_10delta = 
              estimate_mu 1 0 1 obs.x -. estimate_mu 1 0 0 obs.x in
            let mu_00delta = 
              estimate_mu 0 0 1 obs.x -. estimate_mu 0 0 0 obs.x in
            mu_110 +. mu_01delta +. mu_10delta -. mu_00delta)
        |> empirical_mean
      in
      
      (* ATT = E[Y | G = 1, D = 1, T = 1] - correction_term *)
      e_y_treated_post -. correction_term
    
    (* Inverse Propensity Weighting (IPW) based ATT identification *)
    let identify_att_ipw obs_rc pi_estimator =
      let open Utils in
      
      (* Calculate E[G·D·T] *)
      let gdt_array = 
        Array.map (fun o -> float_of_int (o.g * o.d * o.t)) obs_rc in
      let e_gdt = empirical_mean gdt_array in
      
      (* Define φ0(X, G, D, T) function *)
      let phi0 obs pi_fn =
        let x = obs.x in
        let g = obs.g in
        let d = obs.d in
        let t = obs.t in
        
        (* Sum over all (g,d,t) combinations *)
        let sum_term = 
          [|(0,0,0); (0,0,1); (0,1,0); (0,1,1); 
             (1,0,0); (1,0,1); (1,1,0); (1,1,1)|]
          |> Array.map (fun (g',d',t') ->
              let numerator = 
                -1.0 *. float_of_int ((1 - g' - g) * (1 - d' - d) * (1 - t' - t)) in
              let denominator = pi_fn g' d' t' x in
              numerator /. denominator)
          |> Array.fold_left (+.) 0.0
        in
        sum_term
      in
      
      (* Calculate the IPW estimator *)
      let weighted_sum =
        obs_rc
        |> Array.map (fun obs ->
            let y = obs.y in
            let pi_111_x = pi_estimator 1 1 1 obs.x in
            let phi0_val = phi0 obs (fun g d t x -> pi_estimator g d t x) in
            (pi_111_x /. e_gdt) *. phi0_val *. y)
        |> Array.fold_left (+.) 0.0
      in
      
      weighted_sum /. (float_of_int (Array.length obs_rc))
      
    (* Simplified IPW estimator under the no compositional changes assumption *)
    let identify_att_ipw_no_comp_changes obs_rc pi_estimator =
      let open Utils in
      
      (* Calculate E[G·D] and E[T] *)
      let gd_array = Array.map (fun o -> float_of_int (o.g * o.d)) obs_rc in
      let t_array = Array.map (fun o -> float_of_int o.t) obs_rc in
      let e_gd = empirical_mean gd_array in
      let e_t = empirical_mean t_array in
      
      (* Define ρ0(X, G, D) function from panel data setting *)
      let rho0 obs pi_fn =
        let x = obs.x in
        let g = obs.g in
        let d = obs.d in
        
        (* Sum over all (g,d) combinations *)
        let sum_term = 
          [|(0,0); (0,1); (1,0); (1,1)|]
          |> Array.map (fun (g',d') ->
              let numerator = (1 - g' - g) * (1 - d' - d) |> float_of_int in
              let denominator = pi_fn g' d' x in
              numerator /. denominator)
          |> Array.fold_left (+.) 0.0
        in
        sum_term
      in
      
      (* Calculate the simplified IPW estimator *)
      let weighted_sum =
        obs_rc
        |> Array.map (fun obs ->
            let y = obs.y in
            let t_factor = (float_of_int obs.t -. e_t) /. (e_t *. (1.0 -. e_t)) in
            let pi_11_x = pi_estimator 1 1 obs.x in
            let rho0_val = rho0 obs (fun g d x -> pi_estimator g d x) in
            (pi_11_x /. e_gd) *. rho0_val *. t_factor *. y)
        |> Array.fold_left (+.) 0.0
      in
      
      weighted_sum /. (float_of_int (Array.length obs_rc))
end

(* Estimation in the Panel Data Setting *)
module PanelData = struct
    (* Doubly Robust estimator *)
    let estimate_att_dr obs_panel mu_estimator pi_estimator =      
      (* Calculate E[G·D] *)
      let gd_array = Array.map (fun o -> float_of_int (o.g * o.d)) obs_panel in
      let e_gd = empirical_mean gd_array in
      
      (* Define the weight function w_g,d from Equation (8) *)
      let w_gd obs g d =
        let g_obs = obs.g in
        let d_obs = obs.d in
        let x = obs.x in
        let indicator = if g_obs = g && d_obs = d then 1.0 else 0.0 in
        let pi_11_x = pi_estimator 1 1 x in
        let pi_gd_x = pi_estimator g d x in
        
        (float_of_int (g_obs * d_obs) -. pi_11_x *. indicator) /. pi_gd_x
      in
      
      (* Sum over all (g,d) combinations as in Equation (7) *)
      let weighted_sum =
        obs_panel
        |> Array.map (fun obs ->
            let delta_y = obs.y1 -. obs.y0 in
            [|(0,0); (0,1); (1,0); (1,1)|]
            |> Array.map (fun (g,d) ->
                let sign = if (g + d + 1) mod 2 = 0 then 1.0 else -1.0 in
                let weight = w_gd obs g d in
                let mu_gd_x = mu_estimator g d obs.x in
                sign *. weight *. (delta_y -. mu_gd_x))
            |> Array.fold_left (+.) 0.0)
        |> Array.fold_left (+.) 0.0
      in
      
      weighted_sum /. (e_gd *. float_of_int (Array.length obs_panel))
      
    (* Cross-fitting version of the DR estimator to reduce overfitting bias *)
    let estimate_att_dr_crossfit obs_panel mu_estimator pi_estimator k_folds =
      let open Utils in
      let n = Array.length obs_panel in
      
      (* Shuffle the data *)
      let shuffled_indices = Array.init n (fun i -> i) in
      let _ = 
        for i = n - 1 downto 1 do
          let j = Random.int (i + 1) in
          let temp = shuffled_indices.(i) in
          shuffled_indices.(i) <- shuffled_indices.(j);
          shuffled_indices.(j) <- temp
        done
      in
      
      (* Split into k folds *)
      let fold_size = n / k_folds in
      let folds = Array.init k_folds (fun k ->
          let start_idx = k * fold_size in
          let end_idx = min n ((k + 1) * fold_size) in
          Array.init (end_idx - start_idx) (fun i -> 
              shuffled_indices.(start_idx + i))
        )
      in
      
      (* For each fold, train on other folds and predict on the current fold *)
      let fold_estimates = Array.init k_folds (fun k ->
          (* Create training and test sets *)
          let test_indices = folds.(k) in
          let train_indices = 
            Array.init (n - Array.length test_indices) (fun _ -> 0) in
          let train_idx_counter = ref 0 in
          for i = 0 to n - 1 do
            if not (Array.mem i test_indices) then begin
              train_indices.(!train_idx_counter) <- i;
              incr train_idx_counter
            end
          done;
          
          let train_data = Array.map (fun i -> obs_panel.(i)) train_indices in
          let test_data = Array.map (fun i -> obs_panel.(i)) test_indices in
          
          (* Train estimators on training data *)
          let mu_est_k = mu_estimator train_data in
          let pi_est_k = pi_estimator train_data in
          
          (* Apply DR estimator on test data *)
          estimate_att_dr test_data 
            (fun g d x -> mu_est_k g d x)
            (fun g d x -> pi_est_k g d x)
        )
      in
      
      (* Average the fold estimates *)
      Array.fold_left (+.) 0.0 fold_estimates /. float_of_int k_folds
end

(* Estimation in the Repeated Cross-Sections Setting *)
module RepeatedCrossSections = struct
	(* Doubly Robust estimator *)
	let estimate_att_dr obs_rc mu_estimator pi_estimator =

	  (* Calculate E[G·D·T] *)
	  let gdt_array = 
	    Array.map (fun o -> float_of_int (o.g * o.d * o.t)) obs_rc in
	  let e_gdt = empirical_mean gdt_array in
	  
	  (* Define the weight function ω_g,d,t *)
	  let w_gdt obs g d t =
	    let g_obs = obs.g in
	    let d_obs = obs.d in
	    let t_obs = obs.t in
	    let x = obs.x in
	    let indicator = 
	      if g_obs = g && d_obs = d && t_obs = t then 1.0 else 0.0 in
	    let pi_111_x = pi_estimator 1 1 1 x in
	    let pi_gdt_x = pi_estimator g d t x in
	    
	    (float_of_int (g_obs * d_obs * t_obs) -. pi_111_x *. indicator) /. pi_gdt_x
	  in
	  
	  (* Sum over all (g,d,t) combinations *)
	  let weighted_sum =
	    obs_rc
	    |> Array.map (fun obs ->
	        let y = obs.y in
	        [|(0,0,0); (0,0,1); (0,1,0); (0,1,1); (1,0,0); (1,0,1); (1,1,0); (1,1,1)|]
	        |> Array.map (fun (g,d,t) ->
	            let sign = if (g + d + t) mod 2 = 0 then 1.0 else -1.0 in
	            let weight = w_gdt obs g d t in
	            let mu_gdt_x = mu_estimator g d t obs.x in
	            sign *. weight *. (y -. mu_gdt_x))
	        |> Array.fold_left (+.) 0.0)
	    |> Array.fold_left (+.) 0.0
	  in
	  
	  weighted_sum /. (e_gdt *. float_of_int (Array.length obs_rc))
	  
	(* DR estimator under no compositional changes *)
	let estimate_att_dr_no_comp_changes obs_rc mu_estimator pi_estimator =	  
	  (* Calculate E[G·D] and E[T] *)
	  let gd_array = Array.map (fun o -> float_of_int (o.g * o.d)) obs_rc in
	  let t_array = Array.map (fun o -> float_of_int o.t) obs_rc in
	  let e_gd = empirical_mean gd_array in
	  let e_t = empirical_mean t_array in
	  
	  (* Define the weight function w_g,d from panel data setting *)
	  let w_gd obs g d =
	    let g_obs = obs.g in
	    let d_obs = obs.d in
	    let x = obs.x in
	    let indicator = if g_obs = g && d_obs = d then 1.0 else 0.0 in
	    let pi_11_x = pi_estimator 1 1 x in
	    let pi_gd_x = pi_estimator g d x in
	    
	    (float_of_int (g_obs * d_obs) -. pi_11_x *. indicator) /. pi_gd_x
	  in
	  
	  (* Calculate the t-factor *)
	  let t_factor obs =
	    (float_of_int obs.t -. e_t) /. (e_t *. (1.0 -. e_t))
	  in
	  
	  (* Sum over all (g,d) combinations *)
	  let weighted_sum =
	    obs_rc
	    |> Array.map (fun obs ->
	        let y = obs.y in
	        let tf = t_factor obs in
	        [|(0,0); (0,1); (1,0); (1,1)|]
	        |> Array.map (fun (g,d) ->
	            let sign = if (g + d + 1) mod 2 = 0 then 1.0 else -1.0 in
	            let weight = w_gd obs g d in
	            let mu_gd_delta_x = 
	              mu_estimator g d 1 obs.x -. mu_estimator g d 0 obs.x in
	            sign *. weight *. (tf *. y -. mu_gd_delta_x))
	        |> Array.fold_left (+.) 0.0)
	    |> Array.fold_left (+.) 0.0
	  in
	  
	  weighted_sum /. (e_gd *. float_of_int (Array.length obs_rc))
	  
	(* Cross-fitting version of the DR estimator *)
	let estimate_att_dr_crossfit obs_rc mu_estimator pi_estimator k_folds =
	  let open Utils in
	  let n = Array.length obs_rc in
	  
	  (* Shuffle the data *)
	  let shuffled_indices = Array.init n (fun i -> i) in
	  let _ = 
	    for i = n - 1 downto 1 do
	      let j = Random.int (i + 1) in
	      let temp = shuffled_indices.(i) in
	      shuffled_indices.(i) <- shuffled_indices.(j);
	      shuffled_indices.(j) <- temp
	    done
	  in
	  
	  (* Split into k folds *)
	  let fold_size = n / k_folds in
	  let folds = Array.init k_folds (fun k ->
	      let start_idx = k * fold_size in
	      let end_idx = min n ((k + 1) * fold_size) in
	      Array.init (end_idx - start_idx) (fun i -> 
	          shuffled_indices.(start_idx + i))
	    )
	  in
	  
	  (* For each fold, train on other folds and predict on the current fold *)
	  let fold_estimates = Array.init k_folds (fun k ->
	      (* Create training and test sets *)
	      let test_indices = folds.(k) in
	      let train_indices = 
	        Array.init (n - Array.length test_indices) (fun _ -> 0) in
	      let train_idx_counter = ref 0 in
	      for i = 0 to n - 1 do
	        if not (Array.mem i test_indices) then begin
	          train_indices.(!train_idx_counter) <- i;
	          incr train_idx_counter
	        end
	      done;
	      
	      let train_data = Array.map (fun i -> obs_rc.(i)) train_indices in
	      let test_data = Array.map (fun i -> obs_rc.(i)) test_indices in
	      
	      (* Train estimators on training data *)
	      let mu_est_k = mu_estimator train_data in
	      let pi_est_k = pi_estimator train_data in
	      
	      (* Apply DR estimator on test data *)
	      estimate_att_dr test_data 
	        (fun g d t x -> mu_est_k g d t x)
	        (fun g d t x -> pi_est_k g d t x)
	    )
	  in
	  
	  (* Average the fold estimates *)
	  Array.fold_left (+.) 0.0 fold_estimates /. float_of_int k_folds
end

(* Bootstrap confidence intervals for panel data ATT *)
let bootstrap_ci estimator obs_panel n_bootstrap alpha =
    let n = Array.length obs_panel in
    let att_estimates = Array.init n_bootstrap (fun _ ->
        (* Sample with replacement *)
        let bootstrap_sample = Array.init n (fun _ ->
            let idx = Random.int n in
            obs_panel.(idx)
          ) in
        
        (* Compute ATT on bootstrap sample *)
        estimator bootstrap_sample
      ) in
    
    (* Sort the estimates *)
    Array.sort compare att_estimates;
    
    (* Compute confidence interval *)
    let lower_idx = int_of_float (float_of_int n_bootstrap *. alpha /. 2.0) in
    let upper_idx = int_of_float (float_of_int n_bootstrap *. (1.0 -. alpha /. 2.0)) in
    
    (att_estimates.(lower_idx), att_estimates.(upper_idx))
  
(* Bootstrap confidence intervals for repeated cross-sections ATT *)
let bootstrap_ci_rc estimator obs_rc n_bootstrap alpha =
    let n = Array.length obs_rc in
    let att_estimates = Array.init n_bootstrap (fun _ ->
        (* Sample with replacement *)
        let bootstrap_sample = Array.init n (fun _ ->
            let idx = Random.int n in
            obs_rc.(idx)
          ) in
        
        (* Compute ATT on bootstrap sample *)
        estimator bootstrap_sample
      ) in
    
    (* Sort the estimates *)
    Array.sort compare att_estimates;
    
    (* Compute confidence interval *)
    let lower_idx = int_of_float (float_of_int n_bootstrap *. alpha /. 2.0) in
    let upper_idx = int_of_float (float_of_int n_bootstrap *. (1.0 -. alpha /. 2.0)) in
    
    (att_estimates.(lower_idx), att_estimates.(upper_idx))

(* Triple difference estimators *)
module TripleDifference = struct
  (* Panel data triple difference estimator *)
  module PanelData = struct
    (* Configuration type *)
    type config = {
      learning_rate: float;
      batch_size: int;
      epochs: int;
      hidden_dim: int;
      bootstrap_samples: int;
      alpha: float;  (* Significance level for CI, e.g., 0.05 *)
    }
    
    (* Default configuration *)
    let default_config = {
      learning_rate = 0.001;
      batch_size = 32;
      epochs = 100;
      hidden_dim = 64;
      bootstrap_samples = 1000;
      alpha = 0.05;
    }
    
    (* Train outcome regression models *)
    let train_outcome_models ?(config=default_config) obs_panel =
      let n = Array.length obs_panel in
      if n = 0 then Hashtbl.create 0 else begin
        let input_dim = Tensor.shape (Array.get obs_panel 0).x |> List.hd in
        
        (* Create a model for each (g,d) combination *)
        let model_map = Hashtbl.create 4 in
        [|(0,0); (0,1); (1,0); (1,1)|]
        |> Array.iter (fun (g,d) ->
            let model = NeuralOutcomeEstimator.Panel.create 
              ~learning_rate:config.learning_rate
              ~batch_size:config.batch_size
              ~epochs:config.epochs
              ~hidden_dim:config.hidden_dim
              input_dim in
            
            (* Train the model *)
            let trained_model = NeuralOutcomeEstimator.Panel.train model obs_panel g d in
            Hashtbl.add model_map (g,d) trained_model
          );
        
        model_map
      end
    
    (* Train propensity score models *)
    let train_propensity_models ?(config=default_config) obs_panel =
      let n = Array.length obs_panel in
      if n = 0 then Hashtbl.create 0 else begin
        let input_dim = Tensor.shape (Array.get obs_panel 0).x |> List.hd in
        
        (* Create a model for each (g,d) combination *)
        let model_map = Hashtbl.create 4 in
        [|(0,0); (0,1); (1,0); (1,1)|]
        |> Array.iter (fun (g,d) ->
            let model = NeuralPropensityEstimator.Panel.create 
              ~learning_rate:config.learning_rate
              ~batch_size:config.batch_size
              ~epochs:config.epochs
              ~hidden_dim:config.hidden_dim
              input_dim in
            
            (* Train the model *)
            let trained_model = NeuralPropensityEstimator.Panel.train model obs_panel g d in
            Hashtbl.add model_map (g,d) trained_model
          );
        
        model_map
      end
    
    (* Calculate ATT using the simple difference estimator *)
    let simple_diff obs_panel =
      compute_tau_p obs_panel
    
    (* Calculate ATT using outcome regression *)
    let outcome_regression ?(config=default_config) obs_panel =
      (* Train outcome models *)
      let mu_models = train_outcome_models ~config obs_panel in
      let mu_estimator = NeuralOutcomeEstimator.Panel.get_mu_estimator mu_models in
      
      (* Apply OR identification *)
      Identification.identify_att_or obs_panel 
        (fun (g,d) -> mu_estimator g d)
    
    (* Calculate ATT using inverse propensity weighting *)
    let inverse_propensity_weighting ?(config=default_config) obs_panel =
      (* Train propensity models *)
      let pi_models = train_propensity_models ~config obs_panel in
      let pi_estimator = NeuralPropensityEstimator.Panel.get_pi_estimator pi_models in
      
      (* Apply IPW identification *)
      Identification.identify_att_ipw obs_panel 
        (fun g d x -> pi_estimator g d x)
    
    (* Calculate ATT using doubly robust estimator *)
    let doubly_robust ?(config=default_config) obs_panel =
      (* Train outcome and propensity models *)
      let mu_models = train_outcome_models ~config obs_panel in
      let pi_models = train_propensity_models ~config obs_panel in
      
      let mu_estimator = NeuralOutcomeEstimator.Panel.get_mu_estimator mu_models in
      let pi_estimator = NeuralPropensityEstimator.Panel.get_pi_estimator pi_models in
      
      (* Apply DR estimation *)
      Estimation.estimate_att_dr obs_panel 
        (fun g d x -> mu_estimator g d x)
        (fun g d x -> pi_estimator g d x)
    
    (* Calculate ATT with confidence intervals using bootstrap *)
    let doubly_robust_with_ci ?(config=default_config) obs_panel =
      (* Calculate point estimate *)
      let att_point = doubly_robust ~config obs_panel in
      
      (* Calculate confidence interval via bootstrap *)
      let (ci_lower, ci_upper) = bootstrap_ci
        (fun sample -> doubly_robust ~config sample)
        obs_panel
        config.bootstrap_samples
        config.alpha in
      
      (att_point, ci_lower, ci_upper)
    
    (* Calculate ATT using cross-fitting *)
    let doubly_robust_crossfit ?(config=default_config) ?(k_folds=5) obs_panel =
      (* Apply cross-fitting DR estimation *)
      let mu_trainer data =
        let models = train_outcome_models ~config data in
        let estimator = NeuralOutcomeEstimator.Panel.get_mu_estimator models in
        (fun g d x -> estimator g d x)
      in
      
      let pi_trainer data =
        let models = train_propensity_models ~config data in
        let estimator = NeuralPropensityEstimator.Panel.get_pi_estimator models in
        (fun g d x -> estimator g d x)
      in
      
      Estimation.estimate_att_dr_crossfit 
        obs_panel mu_trainer pi_trainer k_folds
  end

  (* Repeated cross-sections triple difference estimator *)
  module RepeatedCrossSections = struct
    (* Configuration type - same as panel data *)
    type config = config
    
    (* Default configuration - same as panel data *)
    let default_config = default_config
    
    (* Train outcome regression models *)
    let train_outcome_models ?(config=default_config) obs_rc =
      let n = Array.length obs_rc in
      if n = 0 then Hashtbl.create 0 else begin
        let input_dim = Tensor.shape (Array.get obs_rc 0).x |> List.hd in
        
        (* Create a model for each (g,d,t) combination *)
        let model_map = Hashtbl.create 8 in
        [|(0,0,0); (0,0,1); (0,1,0); (0,1,1); (1,0,0); (1,0,1); (1,1,0); (1,1,1)|]
        |> Array.iter (fun (g,d,t) ->
            let model = NeuralOutcomeEstimator.RC.create 
              ~learning_rate:config.learning_rate
              ~batch_size:config.batch_size
              ~epochs:config.epochs
              ~hidden_dim:config.hidden_dim
              input_dim in
            
            (* Train the model *)
            let trained_model = NeuralOutcomeEstimator.RC.train model obs_rc g d t in
            Hashtbl.add model_map (g,d,t) trained_model
          );
        
        model_map
      end
    
    (* Train propensity score models *)
    let train_propensity_models ?(config=default_config) obs_rc =
      let n = Array.length obs_rc in
      if n = 0 then Hashtbl.create 0 else begin
        let input_dim = Tensor.shape (Array.get obs_rc 0).x |> List.hd in
        
        (* Create a model for each (g,d,t) combination *)
        let model_map = Hashtbl.create 8 in
        [|(0,0,0); (0,0,1); (0,1,0); (0,1,1); (1,0,0); (1,0,1); (1,1,0); (1,1,1)|]
        |> Array.iter (fun (g,d,t) ->
            let model = NeuralPropensityEstimator.RC.create 
              ~learning_rate:config.learning_rate
              ~batch_size:config.batch_size
              ~epochs:config.epochs
              ~hidden_dim:config.hidden_dim
              input_dim in
            
            (* Train the model *)
            let trained_model = NeuralPropensityEstimator.RC.train model obs_rc g d t in
            Hashtbl.add model_map (g,d,t) trained_model
          );
        
        model_map
      end
    
    (* Train propensity score models for no compositional changes setting *)
    let train_propensity_models_no_comp_changes ?(config=default_config) obs_rc =
      let n = Array.length obs_rc in
      if n = 0 then Hashtbl.create 0 else begin
        let input_dim = Tensor.shape (Array.get obs_rc 0).x |> List.hd in
        
        (* Create a model for each (g,d) combination *)
        let model_map = Hashtbl.create 4 in
        [|(0,0); (0,1); (1,0); (1,1)|]
        |> Array.iter (fun (g,d) ->
            let model = NeuralPropensityEstimator.RC_NoCompChanges.create 
              ~learning_rate:config.learning_rate
              ~batch_size:config.batch_size
              ~epochs:config.epochs
              ~hidden_dim:config.hidden_dim
              input_dim in
            
            (* Train the model using pooled data *)
            let trained_model = 
              NeuralPropensityEstimator.RC_NoCompChanges.train_pooled model obs_rc g d in
            Hashtbl.add model_map (g,d) trained_model
          );
        
        model_map
      end
    
    (* Calculate ATT using the simple difference estimator *)
    let simple_diff obs_rc =
      compute_tau_rc obs_rc
    
    (* Calculate ATT using outcome regression *)
    let outcome_regression ?(config=default_config) obs_rc =
      (* Train outcome models *)
      let mu_models = train_outcome_models ~config obs_rc in
      let mu_estimator = NeuralOutcomeEstimator.RC.get_mu_estimator mu_models in
      
      (* Apply OR identification *)
      Identification.identify_att_or obs_rc 
        (fun (g,d,t) -> mu_estimator g d t)
    
    (* Calculate ATT using inverse propensity weighting *)
    let inverse_propensity_weighting ?(config=default_config) obs_rc =
      (* Train propensity models *)
      let pi_models = train_propensity_models ~config obs_rc in
      let pi_estimator = NeuralPropensityEstimator.RC.get_pi_estimator pi_models in
      
      (* Apply IPW identification *)
      Identification.identify_att_ipw obs_rc 
        (fun g d t x -> pi_estimator g d t x)
    
    (* Calculate ATT using inverse propensity weighting with no compositional changes *)
    let ipw_no_comp_changes ?(config=default_config) obs_rc =
      (* Train propensity models with pooled data *)
      let pi_models = train_propensity_models_no_comp_changes ~config obs_rc in
      let pi_estimator = NeuralPropensityEstimator.Panel.get_pi_estimator pi_models in
      
      (* Apply IPW identification *)
      Identification.identify_att_ipw_no_comp_changes obs_rc 
        (fun g d x -> pi_estimator g d x)
    
    (* Calculate ATT using doubly robust estimator *)
    let doubly_robust ?(config=default_config) obs_rc =
      (* Train outcome and propensity models *)
      let mu_models = train_outcome_models ~config obs_rc in
      let pi_models = train_propensity_models ~config obs_rc in
      
      let mu_estimator = NeuralOutcomeEstimator.RC.get_mu_estimator mu_models in
      let pi_estimator = NeuralPropensityEstimator.RC.get_pi_estimator pi_models in
      
      (* Apply DR estimation *)
      Estimation.estimate_att_dr obs_rc 
        (fun g d t x -> mu_estimator g d t x)
        (fun g d t x -> pi_estimator g d t x)
    
    (* Calculate ATT using doubly robust estimator with no compositional changes *)
    let doubly_robust_no_comp_changes ?(config=default_config) obs_rc =
      (* Train outcome and propensity models *)
      let mu_models = train_outcome_models ~config obs_rc in
      let pi_models = train_propensity_models_no_comp_changes ~config obs_rc in
      
      let mu_estimator = NeuralOutcomeEstimator.RC.get_mu_estimator mu_models in
      let pi_estimator = NeuralPropensityEstimator.Panel.get_pi_estimator pi_models in
      
      (* Apply DR estimation *)
      Estimation.estimate_att_dr_no_comp_changes obs_rc 
        (fun g d t x -> mu_estimator g d t x)
        (fun g d x -> pi_estimator g d x)
    
    (* Calculate ATT with confidence intervals using bootstrap *)
    let doubly_robust_with_ci ?(config=default_config) obs_rc =
      (* Calculate point estimate *)
      let att_point = doubly_robust ~config obs_rc in
      
      (* Calculate confidence interval via bootstrap *)
      let (ci_lower, ci_upper) = bootstrap_ci_rc
        (fun sample -> doubly_robust ~config sample)
        obs_rc
        config.bootstrap_samples
        config.alpha in
      
      (att_point, ci_lower, ci_upper)
    
    (* Calculate ATT using cross-fitting *)
    let doubly_robust_crossfit ?(config=default_config) ?(k_folds=5) obs_rc =
      (* Apply cross-fitting DR estimation *)
      let mu_trainer data =
        let models = train_outcome_models ~config data in
        let estimator = NeuralOutcomeEstimator.RC.get_mu_estimator models in
        (fun g d t x -> estimator g d t x)
      in
      
      let pi_trainer data =
        let models = train_propensity_models ~config data in
        let estimator = NeuralPropensityEstimator.RC.get_pi_estimator pi_models in
        (fun g d t x -> estimator g d t x)
      in
      
      Estimation.estimate_att_dr_crossfit 
        obs_rc mu_trainer pi_trainer k_folds
  end
end

(* Monte Carlo simulation for evaluating the estimators *)
module MonteCarlo = struct
  (* Simulation parameters *)
  type sim_params = {
    n_units: int;
    dim_x: int;
    true_att: float;
    n_simulations: int;
    seed: int;
  }
  
  (* Default simulation parameters *)
  let default_params = {
    n_units = 1000;
    dim_x = 5;
    true_att = 2.0;
    n_simulations = 100;
    seed = 42;
  }
  
  (* Generate synthetic panel data for simulations *)
  let generate_panel_data params =
    (* Set random seed for reproducibility *)
    Random.init params.seed;
    
    (* Initialize synthetic data *)
    Array.init params.n_units (fun _ ->
      (* Generate G and D indicators *)
      let g = if Random.float 1.0 < 0.5 then 1 else 0 in
      let d = if Random.float 1.0 < 0.5 then 1 else 0 in
      
      (* Generate covariates *)
      let x = Tensor.randn [params.dim_x] ~mean:0.0 ~std:1.0 ~device:Device.Cpu in
      
      (* Generate potential outcomes *)
      (* Baseline outcome affected by covariates *)
      let baseline = 
        let x_sum = 
          let x_float = Utils.float_array_of_tensor x in
          Array.fold_left (+.) 0.0 x_float
        in
        x_sum /. float_of_int params.dim_x
      in
      
      (* Time trend common to all units *)
      let time_trend = 0.5 in
      
      (* Group-specific effects *)
      let g_effect = if g = 1 then 1.0 else 0.0 in
      let d_effect = if d = 1 then 0.5 else 0.0 in
      
      (* Heterogeneity in trends based on group and domain *)
      let g_trend = if g = 1 then 0.3 else 0.0 in
      let d_trend = if d = 1 then 0.2 else 0.0 in
      
      (* Treatment effect only for treated group (G=1, D=1) *)
      let treatment_effect = if g = 1 && d = 1 then params.true_att else 0.0 in
      
      (* Add random noise *)
      let noise0 = Random.float 1.0 -. 0.5 in
      let noise1 = Random.float 1.0 -. 0.5 in
      
      (* Final outcomes *)
      let y0 = baseline +. g_effect +. d_effect +. noise0 in
      let y1 = y0 +. time_trend +. g_trend +. d_trend +. treatment_effect +. noise1 in
      
      { g; d; x; y0; y1 }
    )
  
  (* Generate synthetic repeated cross-sections data for simulations *)
  let generate_rc_data params =
    (* Set random seed for reproducibility *)
    Random.init params.seed;
    
    (* Initialize synthetic data *)
    Array.init (params.n_units * 2) (fun i ->
      (* Generate G, D, and T indicators *)
      let t = if i < params.n_units then 0 else 1 in
      let g = if Random.float 1.0 < 0.5 then 1 else 0 in
      let d = if Random.float 1.0 < 0.5 then 1 else 0 in
      
      (* Generate covariates *)
      let x = Tensor.randn [params.dim_x] ~mean:0.0 ~std:1.0 ~device:Device.Cpu in
      
      (* Generate outcomes *)
      (* Baseline outcome affected by covariates *)
      let baseline = 
        let x_sum = 
          let x_float = Utils.float_array_of_tensor x in
          Array.fold_left (+.) 0.0 x_float
        in
        x_sum /. float_of_int params.dim_x
      in
      
      (* Time trend common to all units *)
      let time_trend = if t = 1 then 0.5 else 0.0 in
      
      (* Group-specific effects *)
      let g_effect = if g = 1 then 1.0 else 0.0 in
      let d_effect = if d = 1 then 0.5 else 0.0 in
      
      (* Heterogeneity in trends based on group and domain *)
      let g_trend = if g = 1 && t = 1 then 0.3 else 0.0 in
      let d_trend = if d = 1 && t = 1 then 0.2 else 0.0 in
      
      (* Treatment effect only for treated post-treatment group (G=1, D=1, T=1) *)
      let treatment_effect = if g = 1 && d = 1 && t = 1 then params.true_att else 0.0 in
      
      (* Add random noise *)
      let noise = Random.float 1.0 -. 0.5 in
      
      (* Final outcome *)
      let y = baseline +. g_effect +. d_effect +. time_trend +. g_trend +. d_trend +. treatment_effect +. noise in
      
      { g; d; t; x; y }
    )
  
  (* Run panel data simulations *)
  let run_panel_simulations ?(params=default_params) ?(config=TripleDifference.default_config) () =
    let n_sims = params.n_simulations in
    
    (* Storage for results *)
    let results_simple = Array.make n_sims 0.0 in
    let results_or = Array.make n_sims 0.0 in
    let results_ipw = Array.make n_sims 0.0 in
    let results_dr = Array.make n_sims 0.0 in
    
    (* Run simulations *)
    for i = 0 to n_sims - 1 do
      (* Generate data for this simulation *)
      let seed = params.seed + i in
      let sim_params = { params with seed } in
      let data = generate_panel_data sim_params in
      
      (* Run estimators *)
      results_simple.(i) <- TripleDifference.simple_diff data;
      results_or.(i) <- TripleDifference.outcome_regression ~config data;
      results_ipw.(i) <- TripleDifference.inverse_propensity_weighting ~config data;
      results_dr.(i) <- TripleDifference.doubly_robust ~config data;
      
      (* Print progress *)
      if (i + 1) mod 10 = 0 then
        Printf.printf "Completed simulation %d/%d\n" (i + 1) n_sims;
    done;
    
    (* Calculate bias, variance, and MSE *)
    let calculate_stats results =
      let n = Array.length results in
      let true_att = params.true_att in
      
      (* Mean estimate *)
      let mean_est = Utils.mean_float results in
      
      (* Bias *)
      let bias = mean_est -. true_att in
      
      (* Variance *)
      let var = 
        Array.fold_left (fun acc x -> acc +. ((x -. mean_est) ** 2.0)) 0.0 results
        /. float_of_int n in
      
      (* MSE *)
      let mse = 
        Array.fold_left (fun acc x -> acc +. ((x -. true_att) ** 2.0)) 0.0 results
        /. float_of_int n in
      
      (bias, var, mse)
    in
    
    let (bias_simple, var_simple, mse_simple) = calculate_stats results_simple in
    let (bias_or, var_or, mse_or) = calculate_stats results_or in
    let (bias_ipw, var_ipw, mse_ipw) = calculate_stats results_ipw in
    let (bias_dr, var_dr, mse_dr) = calculate_stats results_dr in
    
    (* Print results *)
    Printf.printf "\nPanel Data Simulation Results (True ATT = %.2f):\n" params.true_att;
    Printf.printf "Estimator\tBias\tVariance\tMSE\n";
    Printf.printf "Simple\t\t%.4f\t%.4f\t\t%.4f\n" bias_simple var_simple mse_simple;
    Printf.printf "OR\t\t%.4f\t%.4f\t\t%.4f\n" bias_or var_or mse_or;
    Printf.printf "IPW\t\t%.4f\t%.4f\t\t%.4f\n" bias_ipw var_ipw mse_ipw;
    Printf.printf "DR\t\t%.4f\t%.4f\t\t%.4f\n" bias_dr var_dr mse_dr;
    
    (results_simple, results_or, results_ipw, results_dr)
  
  (* Run repeated cross-sections simulations *)
  let run_rc_simulations ?(params=default_params) ?(config=TripleDifference.default_config) ?(assume_no_comp_changes=false) () =
    let n_sims = params.n_simulations in
    
    (* Storage for results *)
    let results_simple = Array.make n_sims 0.0 in
    let results_or = Array.make n_sims 0.0 in
    let results_ipw = Array.make n_sims 0.0 in
    let results_dr = Array.make n_sims 0.0 in
    
    (* For no compositional changes assumption *)
    let results_ipw_nc = Array.make n_sims 0.0 in
    let results_dr_nc = Array.make n_sims 0.0 in
    
    (* Run simulations *)
    for i = 0 to n_sims - 1 do
      (* Generate data for this simulation *)
      let seed = params.seed + i in
      let sim_params = { params with seed } in
      let data = generate_rc_data sim_params in
      
      (* Run estimators *)
      results_simple.(i) <- TripleDifference.simple_diff data;
      results_or.(i) <- TripleDifference.outcome_regression ~config data;
      results_ipw.(i) <- TripleDifference.inverse_propensity_weighting ~config data;
      results_dr.(i) <- TripleDifference.doubly_robust ~config data;
      
      (* Run no compositional changes estimators *)
      if assume_no_comp_changes then begin
        results_ipw_nc.(i) <- TripleDifference.ipw_no_comp_changes ~config data;
        results_dr_nc.(i) <- TripleDifference.doubly_robust_no_comp_changes ~config data;
      end;
      
      (* Print progress *)
      if (i + 1) mod 10 = 0 then
        Printf.printf "Completed simulation %d/%d\n" (i + 1) n_sims;
    done;
    
    (* Calculate bias, variance, and MSE *)
    let calculate_stats results =
      let n = Array.length results in
      let true_att = params.true_att in
      
      (* Mean estimate *)
      let mean_est = Utils.mean_float results in
      
      (* Bias *)
      let bias = mean_est -. true_att in
      
      (* Variance *)
      let var = 
        Array.fold_left (fun acc x -> acc +. ((x -. mean_est) ** 2.0)) 0.0 results
        /. float_of_int n in
      
      (* MSE *)
      let mse = 
        Array.fold_left (fun acc x -> acc +. ((x -. true_att) ** 2.0)) 0.0 results
        /. float_of_int n in
      
      (bias, var, mse)
    in
    
    let (bias_simple, var_simple, mse_simple) = calculate_stats results_simple in
    let (bias_or, var_or, mse_or) = calculate_stats results_or in
    let (bias_ipw, var_ipw, mse_ipw) = calculate_stats results_ipw in
    let (bias_dr, var_dr, mse_dr) = calculate_stats results_dr in
    
    (* Print results *)
    Printf.printf "\nRepeated Cross-Sections Simulation Results (True ATT = %.2f):\n" params.true_att;
    Printf.printf "Estimator\tBias\tVariance\tMSE\n";
    Printf.printf "Simple\t\t%.4f\t%.4f\t\t%.4f\n" bias_simple var_simple mse_simple;
    Printf.printf "OR\t\t%.4f\t%.4f\t\t%.4f\n" bias_or var_or mse_or;
    Printf.printf "IPW\t\t%.4f\t%.4f\t\t%.4f\n" bias_ipw var_ipw mse_ipw;
    Printf.printf "DR\t\t%.4f\t%.4f\t\t%.4f\n" bias_dr var_dr mse_dr;
    
    (* Print no compositional changes results if applicable *)
    if assume_no_comp_changes then begin
      let (bias_ipw_nc, var_ipw_nc, mse_ipw_nc) = calculate_stats results_ipw_nc in
      let (bias_dr_nc, var_dr_nc, mse_dr_nc) = calculate_stats results_dr_nc in
      Printf.printf "\nNo Compositional Changes Assumption Results:\n";
      Printf.printf "IPW-NC\t\t%.4f\t%.4f\t\t%.4f\n" bias_ipw_nc var_ipw_nc mse_ipw_nc;
      Printf.printf "DR-NC\t\t%.4f\t%.4f\t\t%.4f\n" bias_dr_nc var_dr_nc mse_dr_nc;
      
      (results_simple, results_or, results_ipw, results_dr, results_ipw_nc, results_dr_nc)
    end else
      (results_simple, results_or, results_ipw, results_dr, [||], [||])
  
  (* Compare estimators under different sample sizes *)
  let sample_size_sensitivity ?(base_params=default_params) ?(config=TripleDifference.default_config) sample_sizes =
    (* For each sample size, run simulations and collect MSE *)
    let results = Array.map (fun n ->
        let params = { base_params with n_units = n; n_simulations = 20 } in
        let (_, _, _, results_dr) = run_panel_simulations ~params ~config () in
        
        (* Calculate MSE *)
        let true_att = params.true_att in
        let mse = 
          Array.fold_left (fun acc x -> acc +. ((x -. true_att) ** 2.0)) 0.0 results_dr
          /. float_of_int (Array.length results_dr) in
        
        (n, mse)
      ) sample_sizes in
    
    (* Print results *)
    Printf.printf "\nSample Size Sensitivity Analysis:\n";
    Printf.printf "Sample Size\tMSE\n";
    Array.iter (fun (n, mse) -> Printf.printf "%d\t\t%.4f\n" n mse) results;
    
    results
  
  (* Compare estimators under different treatment effect sizes *)
  let effect_size_sensitivity ?(base_params=default_params) ?(config=TripleDifference.default_config) effect_sizes =
    (* For each effect size, run simulations and collect bias *)
    let results = Array.map (fun effect ->
        let params = { base_params with true_att = effect; n_simulations = 20 } in
        let (_, _, _, results_dr) = run_panel_simulations ~params ~config () in
        
        (* Calculate bias *)
        let mean_est = Utils.mean_float results_dr in
        let bias = mean_est -. effect in
        
        (effect, bias)
      ) effect_sizes in
    
    (* Print results *)
    Printf.printf "\nEffect Size Sensitivity Analysis:\n";
    Printf.printf "Effect Size\tBias\n";
    Array.iter (fun (effect, bias) -> Printf.printf "%.2f\t\t%.4f\n" effect bias) results;
    
    results
end