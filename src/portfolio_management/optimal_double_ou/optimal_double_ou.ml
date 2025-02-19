open Torch

type predictor = {
  epsilon: float;      (* Mean reversion rate *)
  psi: float;         (* Volatility *)
  beta: float;        (* Market beta *)
  max_pos: float;     (* Maximum position *)
  gamma: float;       (* Individual cost parameter *)
}

type trading_params = {
  dt: float;                (* Time step *)
  gamma: float;             (* Global cost parameter *)
  lambda: float;            (* Risk aversion *)
  num_assets: int;          (* Number of assets *)
  max_risk: float;          (* Maximum allowed risk *)
  max_cost: float;          (* Maximum allowed cost *)
  initial_capital: float;   (* Initial capital *)
  predictors: predictor array;  (* Array of predictors *)
}

type position = {
  value: float;
  asset_idx: int;
}

let adaptive_integrate ?(tol=1e-6) ?(max_steps=1000) f a b =
  let rec integrate_step a b f_a f_b acc step =
    if step >= max_steps then acc
    else
      let mid = (a +. b) /. 2.0 in
      let f_mid = f mid in
      let area1 = (b -. a) *. (f_a +. f_b) /. 2.0 in
      let area2 = (b -. a) *. (f_a +. 2.0 *. f_mid +. f_b) /. 4.0 in
      
      if abs_float (area1 -. area2) < tol *. abs_float area2 then
        acc +. area2
      else
        integrate_step a mid f_a f_mid 
          (integrate_step mid b f_mid f_b acc (step + 1)) 
          (step + 1)
  in
  integrate_step a b (f a) (f b) 0.0 0

let stabilize_covariance matrix =
  let n = Array.length matrix in
  let result = Array.make_matrix n n 0.0 in
  
  for i = 0 to n - 1 do
    for j = i to n - 1 do
      let avg = (matrix.(i).(j) +. matrix.(j).(i)) /. 2.0 in
      result.(i).(j) <- avg;
      result.(j).(i) <- avg;
      
      if i = j then
        result.(i).(i) <- max result.(i).(i) 1e-8
    done
  done;
  result

(* Ornstein-Uhlenbeck process *)
module OrnsteinUhlenbeck = struct
  type state = {
    value: Tensor.t;
    time: float;
    path_max: float;
    path_min: float;
    crossings: int;
  }

  type regime = 
    | Weak of float       (* ψ ≪ Γε^(3/2) *)
    | Intermediate of float  (* Γε^(3/2) ≪ ψ ≪ Γ *)
    | Strong of float     (* ψ ≳ Γ *)

  let create initial_value =
    {
      value = Tensor.float_tensor [initial_value];
      time = 0.0;
      path_max = initial_value;
      path_min = initial_value;
      crossings = 0;
    }

  let identify_regime pred params =
    let epsilon = pred.epsilon in
    let psi = pred.psi in
    let gamma = params.gamma in
    
    let weak_bound = gamma *. epsilon ** 1.5 in
    let strong_bound = gamma in
    
    if psi < weak_bound then
      Weak(psi /. weak_bound)
    else if psi < strong_bound then
      Intermediate((psi -. weak_bound) /. (strong_bound -. weak_bound))
    else
      Strong(psi /. strong_bound)

  let update_exact pred state dt =
    let mean = Tensor.mul_scalar state.value (exp (-. pred.epsilon *. dt)) in
    let var = pred.psi *. pred.psi *. 
      (1.0 -. exp (-2.0 *. pred.epsilon *. dt)) /. (2.0 *. pred.epsilon) in
    let noise = Tensor.normal ~mean:0.0 ~std:(sqrt var) [1] ~device:Device.Cpu in
    let new_value = Tensor.add mean noise in
    
    (* Track path properties *)
    let value = Tensor.float_value new_value in
    let max_val = max state.path_max value in
    let min_val = min state.path_min value in
    let crossings = if value *. Tensor.float_value state.value < 0.0 then
      state.crossings + 1
    else
      state.crossings in
    
    {
      value = new_value;
      time = state.time +. dt;
      path_max = max_val;
      path_min = min_val;
      crossings;
    }

  let simulate_path ?(n_steps=1000) ?(dt=0.01) pred init_value =
    let state = create init_value in
    let states = Array.make (n_steps + 1) state in
    
    for i = 1 to n_steps do
      states.(i) <- update_exact pred states.(i-1) dt
    done;
    
    states

  let stationary_density pred x =
    let var = pred.psi *. pred.psi /. (2.0 *. pred.epsilon) in
    1.0 /. sqrt (2.0 *. Float.pi *. var) *. 
    exp (-. x *. x /. (2.0 *. var))
end

(* Risk calculation and management *)
module RiskCalculation = struct
  type risk_decomposition = {
    total_risk: float;
    systematic_risk: float;
    specific_risk: float;
    risk_contributions: float array;
  }

  let calculate_risk_decomposition positions predictors =
    let n = Array.length predictors in
    
    (* Calculate specific risk *)
    let specific_risk = Array.fold_left2 (fun acc pos pred ->
      let var = pred.psi *. pred.psi /. (2.0 *. pred.epsilon) in
      acc +. pos *. pos *. var
    ) 0.0 positions predictors in
    
    (* Calculate systematic risk *)
    let market_exposure = Array.fold_left2 (fun acc pos pred ->
      acc +. pos *. pred.beta
    ) 0.0 positions predictors in
    let systematic_risk = market_exposure *. market_exposure in
    
    (* Calculate risk contributions *)
    let risk_contributions = Array.mapi (fun i pos ->
      let pred = predictors.(i) in
      let spec_risk = pos *. pos *. pred.psi *. pred.psi /. 
        (2.0 *. pred.epsilon) in
      let sys_risk = pos *. pred.beta *. market_exposure in
      spec_risk +. sys_risk
    ) positions in
    
    {
      total_risk = sqrt (specific_risk +. systematic_risk);
      systematic_risk = sqrt systematic_risk;
      specific_risk = sqrt specific_risk;
      risk_contributions;
    }

  let calculate_min_risk predictors =
    let n = Array.length predictors in
    
    (* Calculate specific risk lower bound *)
    let specific_risk = Array.fold_left (fun acc pred ->
      acc +. (pred.psi *. pred.psi /. (2.0 *. pred.epsilon)) *. 
        pred.max_pos *. pred.max_pos
    ) 0.0 predictors /. float n in
    
    (* Sort betas for optimal allocation *)
    let sorted_betas = Array.mapi (fun i pred -> 
      (pred.beta, pred.max_pos, i)
    ) predictors |> Array.copy in
    Array.sort (fun (b1,_,_) (b2,_,_) -> compare b2 b1) sorted_betas;
    
    (* Calculate minimum systematic risk *)
    let systematic_risk = Array.fold_left (fun acc (beta, max_pos, _) ->
      acc +. beta *. beta *. max_pos *. max_pos
    ) 0.0 (Array.sub sorted_betas 0 (min n 10)) /. float n in
    
    sqrt (specific_risk +. systematic_risk)
end

let calculate_threshold pred params =
  match OrnsteinUhlenbeck.identify_regime pred params with
  | OrnsteinUhlenbeck.Weak(_) ->
      params.gamma *. pred.epsilon
  | OrnsteinUhlenbeck.Intermediate(_) ->
      (1.5 *. params.gamma *. pred.psi *. pred.psi) ** (1.0 /. 3.0)
  | OrnsteinUhlenbeck.Strong(_) ->
      params.gamma

let calculate_modified_threshold pred risk theta =
  let base_threshold = calculate_threshold pred in
  let p_star = pred.psi /. sqrt pred.epsilon in
  let q_hat = base_threshold /. p_star in
  
  (* Calculate slope based on regime *)
  let slope = match OrnsteinUhlenbeck.identify_regime pred with
  | OrnsteinUhlenbeck.Weak(_) ->
      1.0  (* No significant adjustment in weak regime *)
  | OrnsteinUhlenbeck.Intermediate(_) ->
      let term1 = 1.0 -. q_hat *. q_hat in
      let term2 = 1.0 +. 2.0 *. q_hat *. q_hat /. pred.epsilon in
      1.0 /. (term1 *. term2)
  | OrnsteinUhlenbeck.Strong(_) ->
      0.5  (* Reduced adjustment in strong regime *)
  in
  
  base_threshold +. slope *. theta *. risk

let calculate_trading_rate pred threshold =
  let p_star = pred.psi /. sqrt pred.epsilon in
  let q_hat = threshold /. p_star in
  
  if q_hat < 1.0 then
    pred.epsilon /. (2.0 *. sqrt Float.pi) /. q_hat
  else
    sqrt pred.epsilon /. sqrt Float.pi *. exp (-. q_hat *. q_hat)

let calculate_costs params new_positions old_positions =
  Array.fold_left2 (fun acc new_pos old_pos ->
    acc +. params.gamma *. abs_float (new_pos -. old_pos)
  ) 0.0 new_positions old_positions

let optimize_trade params pred curr_pos target_pos =
  let direction = if target_pos > curr_pos then 1.0 else -1.0 in
  let distance = abs_float (target_pos -. curr_pos) in
  
  match OrnsteinUhlenbeck.identify_regime pred params with
  | OrnsteinUhlenbeck.Weak(_) ->
      (* Trade slowly in weak regime *)
      if distance < params.gamma then target_pos
      else curr_pos +. direction *. params.gamma
  | OrnsteinUhlenbeck.Intermediate(_) ->
      (* Trade moderately in intermediate regime *)
      if distance < 2.0 *. params.gamma then target_pos
      else curr_pos +. direction *. params.gamma *. 2.0
  | OrnsteinUhlenbeck.Strong(_) ->
      (* Trade aggressively in strong regime *)
      target_pos

let optimize_positions params predictors curr_positions =
  let horizon = 20 in
  let opt_positions = optimize_trajectory params predictors curr_positions horizon in
  
  (* Risk adjustment *)
  let risk_decomp = RiskCalculation.calculate_risk_decomposition 
    opt_positions predictors in
  
  if risk_decomp.total_risk > params.max_risk then
    (* Scale back positions to meet risk constraint *)
    let scale = sqrt (params.max_risk /. risk_decomp.total_risk) in
    Array.map (fun pos -> pos *. scale) opt_positions
  else
    opt_positions

(* Trading system *)
module TradingSystem = struct
  type system_state = {
    positions: float array;
    predictors: float array;
    risk: float;
    costs: float;
    pnl: float;
    time: float;
    regime_states: OrnsteinUhlenbeck.regime array;
  }

  let create_system params =
    let n = Array.length params.predictors in
    {
      positions = Array.make n 0.0;
      predictors = Array.make n 0.0;
      risk = 0.0;
      costs = 0.0;
      pnl = 0.0;
      time = 0.0;
      regime_states = Array.map (fun pred -> 
        OrnsteinUhlenbeck.identify_regime pred params
      ) params.predictors;
    }

  let update_system params state =
    let n = Array.length params.predictors in
    
    (* Update predictors *)
    let new_predictors = Array.mapi (fun i pred ->
      let curr_pred = state.predictors.(i) in
      let noise = Random.float 2.0 -. 1.0 in
      curr_pred -. pred.epsilon *. curr_pred *. params.dt +.
      pred.psi *. noise *. sqrt params.dt
    ) params.predictors in
    
    (* Optimize positions *)
    let new_positions = optimize_positions 
      params new_predictors state.positions in
    
    (* Calculate trading costs *)
    let step_costs = calculate_costs 
      params new_positions state.positions in
    
    (* Calculate risk decomposition *)
    let risk_decomp = RiskCalculation.calculate_risk_decomposition 
      new_positions params.predictors in
    
    (* Calculate PnL *)
    let step_pnl = Array.fold_left2 (fun acc pos pred ->
      acc +. pos *. pred
    ) 0.0 new_positions new_predictors in
    
    (* Update regime states *)
    let new_regimes = Array.map (fun pred ->
      OrnsteinUhlenbeck.identify_regime pred params
    ) params.predictors in
    
    {
      positions = new_positions;
      predictors = new_predictors;
      risk = risk_decomp.total_risk;
      costs = state.costs +. step_costs;
      pnl = state.pnl +. step_pnl -. step_costs;
      time = state.time +. params.dt;
      regime_states = new_regimes;
    }

  let run_simulation params init_state n_steps =
    let states = Array.make (n_steps + 1) init_state in
    
    for t = 1 to n_steps do
      states.(t) <- update_system params states.(t-1)
    done;
    
    states
end

(* Mean field dynamics *)
module MeanFieldDynamics = struct
  type field_state = {
    positions: float array;
    predictors: float array;
    risk: float;
    trading_rates: float array;
  }

  let create_state n =
    {
      positions = Array.make n 0.0;
      predictors = Array.make n 0.0;
      risk = 0.0;
      trading_rates = Array.make n 0.0;
    }

  let calculate_trading_rate pred pos =
    match OrnsteinUhlenbeck.identify_regime pred with
    | OrnsteinUhlenbeck.Weak(_) ->
        pred.epsilon /. (2.0 *. sqrt Float.pi)
    | OrnsteinUhlenbeck.Intermediate(_) ->
        let p_star = pred.psi /. sqrt pred.epsilon in
        pred.epsilon /. (2.0 *. sqrt Float.pi) /. (p_star /. pred.max_pos)
    | OrnsteinUhlenbeck.Strong(_) ->
        sqrt pred.epsilon /. sqrt Float.pi

  let update_dynamics params state dt =
    let n = Array.length params.predictors in
    
    (* Update predictors *)
    let new_predictors = Array.mapi (fun i pred ->
      let curr_pred = state.predictors.(i) in
      let noise = Random.float 2.0 -. 1.0 in
      curr_pred -. pred.epsilon *. curr_pred *. dt +.
      pred.psi *. noise *. sqrt dt
    ) params.predictors in
    
    (* Calculate trading rates *)
    let new_rates = Array.mapi (fun i pred ->
      calculate_trading_rate pred state.positions.(i)
    ) params.predictors in
    
    (* Update risk process *)
    let j_bar = Array.fold_left2 (fun acc rate pred ->
      acc +. rate *. pred.beta *. pred.beta
    ) 0.0 new_rates params.predictors /. float n in
    
    let sigma = sqrt (Array.fold_left (fun acc pred ->
      acc +. pred.beta *. pred.beta *. pred.max_pos *. pred.max_pos
    ) 0.0 params.predictors /. float n) in
    
    let new_risk = state.risk -. 
      2.0 *. j_bar *. state.risk *. dt +.
      2.0 *. sigma *. sqrt (j_bar *. dt) *. (Random.float 2.0 -. 1.0) in
    
    (* Update positions *)
    let new_positions = Array.mapi (fun i pred ->
      let curr_pos = state.positions.(i) in
      let threshold = calculate_modified_threshold 
        pred new_risk (params.lambda *. pred.beta /. sqrt (float n)) in
      
      if new_predictors.(i) >= threshold then pred.max_pos
      else if new_predictors.(i) <= -.threshold then -.pred.max_pos
      else curr_pos
    ) params.predictors in
    
    {
      positions = new_positions;
      predictors = new_predictors;
      risk = new_risk;
      trading_rates = new_rates;
    }
end