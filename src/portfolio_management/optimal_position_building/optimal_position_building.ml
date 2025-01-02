open Torch

type strategy = float -> float
type multi_asset_strategy = int -> float -> float
type loss_function = float -> float -> float -> float

type market_model = {
  price_process: float -> float -> float;
  volatility: float -> float;
  liquidity: float -> float;
}

type regime = Bull | Bear | Sideways | Volatile

type risk_measure = 
  | ValueAtRisk of float
  | ExpectedShortfall of float
  | MaxDrawdown

type performance_metric = 
  | SharpeRatio
  | SortinoRatio
  | InformationRatio of strategy
  | CalmarRatio

type probability_measure = float -> float

type optimization_constraint =
  | MaxWeight of float
  | MinWeight of float
  | SectorExposure of int array * float * float

type market_impact_model =
  | Linear of float
  | SquareRoot of float
  | PowerLaw of float * float
  | TempermantonPermanent of float * float

type factor = {
  name: string;
  beta: float;
  returns: float array;
}

type objective_function = Tensor.t -> float
type constraint_function = Tensor.t -> bool

type ensemble_method = 
  | EqualWeight
  | InverseVolatility
  | OptimalF
  | KellyWeights

type market_regime = Bull | Bear | Sideways | Volatile

type var_method = Historical | Parametric | MonteCarloVaR
type risk_metric = 
  | ValueAtRisk of float * var_method
  | ConditionalVaR of float * var_method
  | ExpectedShortfall of float
  | DownsideDeviation of float

type simulation_state = {
  time: float;
  prices: float array;
  positions: float array;
  cash: float;
}

type portfolio = {
  weights: float array;
  last_rebalance: float;
}

type backtest_result = {
  returns: float array;
  sharpe_ratio: float;
  max_drawdown: float;
  total_pnl: float;
  total_cost: float;
}

type advanced_backtest_result = {
  base_result: backtest_result;
  var: float;
  cvar: float;
  calmar_ratio: float;
  omega_ratio: float;
  sortino_ratio: float;
}

let sigmoid x = 1. /. (1. +. exp (-.x))

let risk_neutral_strategy t = t

let risk_averse_strategy kappa t =
  (exp (kappa *. t) -. 1.) /. (exp kappa -. 1.)

let eager_strategy alpha t =
  (exp (alpha *. t) -. 1.) /. (exp alpha -. 1.)

let compute_cost a b lambda kappa gamma =
  let open Tensor in
  let t = linspace ~start:0. ~end_:1. ~steps:1000 in
  let a_t = of_float1 (Array.map a (to_float1_exn t)) in
  let b_t = of_float1 (Array.map b (to_float1_exn t)) in
  let da_dt = diff a_t ~dim:0 ~n:1 in
  let db_dt = diff b_t ~dim:0 ~n:1 in
  let temp_impact = mul (add da_dt db_dt) da_dt in
  let perm_impact = mul (mul (add a_t b_t) da_dt) (float gamma) in
  let total_cost = add temp_impact perm_impact in
  sum total_cost |> to_float0_exn

let geometric_brownian_motion mu sigma =
  let price_process s0 t =
    let z = Tensor.randn [1] |> Tensor.to_float0_exn in
    s0 *. exp((mu -. 0.5 *. sigma ** 2.) *. t +. sigma *. sqrt t *. z)
  in
  let volatility _ = sigma in
  let liquidity _ = 1.0 in
  { price_process; volatility; liquidity }

let jump_diffusion_model mu sigma lambda jump_size =
  let price_process s0 t =
    let z = Tensor.randn [1] |> Tensor.to_float0_exn in
    let n = Tensor.poisson ~lambda:(lambda *. t) [1] |> Tensor.to_int0_exn in
    let jumps = List.init n (fun _ -> Tensor.randn [1] |> Tensor.to_float0_exn) |> List.fold_left (+.) 0. in
    s0 *. exp((mu -. 0.5 *. sigma ** 2.) *. t +. sigma *. sqrt t *. z +. float n *. log(1. +. jump_size) +. jump_size *. jumps)
  in
  let volatility t = sqrt (sigma ** 2. +. lambda *. jump_size ** 2. *. t) in
  let liquidity _ = 1.0 in
  { price_process; volatility; liquidity }

let stochastic_volatility_model mu kappa theta sigma =
  let price_process s0 t =
    let v0 = theta in
    let z1 = Tensor.randn [1] |> Tensor.to_float0_exn in
    let z2 = Tensor.randn [1] |> Tensor.to_float0_exn in
    let v_t = v0 *. exp(-kappa *. t) +. theta *. (1. -. exp(-kappa *. t)) +. 
              sigma *. sqrt((1. -. exp(-2. *. kappa *. t)) /. (2. *. kappa)) *. z1 in
    s0 *. exp((mu -. 0.5 *. v_t) *. t +. sqrt v_t *. sqrt t *. z2)
  in
  let volatility t =
    let v0 = theta in
    sqrt (v0 *. exp(-kappa *. t) +. theta *. (1. -. exp(-kappa *. t)))
  in
  let liquidity _ = 1.0 in
  { price_process; volatility; liquidity }

let create_regime_switching_model regimes transition_matrix model_params =
  let price_process s0 t =
    let current_regime = ref 0 in
    let price = ref s0 in
    for _ = 1 to int_of_float (t *. 252.) do
      let (mu, sigma, jump_intensity) = model_params.(!current_regime) in
      let z = Tensor.randn [1] |> Tensor.to_float0_exn in
      let jump = if Random.float 1.0 < jump_intensity then Tensor.randn [1] |> Tensor.to_float0_exn else 0.0 in
      price := !price *. exp((mu -. 0.5 *. sigma ** 2.) +. sigma *. z +. jump);
      let r = Random.float 1.0 in
      let cumulative_prob = ref 0.0 in
      for j = 0 to Array.length regimes - 1 do
        cumulative_prob := !cumulative_prob +. transition_matrix.(!current_regime).(j);
        if r < !cumulative_prob then (
          current_regime := j;
          cumulative_prob := 1.0
        )
      done
    done;
    !price
  in
  let volatility _ = model_params.(!0).(1) in
  let liquidity _ = 1.0 in
  { price_process; volatility; liquidity }

let simulate_regime_switching_price model initial_price num_steps =
  Array.init num_steps (fun i -> model.price_process initial_price (float i /. float num_steps))

(* Equilibrium Strategies *)
let best_response_strategy b lambda kappa t =
  let open Tensor in
  let t_tensor = float t |> of_float0 in
  let b_t = float (b t) |> of_float0 in
  let db_dt = (b (t +. 1e-6) -. b (t -. 1e-6)) /. 2e-6 |> of_float0 in
  let d2b_dt2 = ((b (t +. 1e-6) -. 2. *. b t +. b (t -. 1e-6)) /. (1e-6 ** 2.)) |> of_float0 in
  
  let numerator = neg (mul (float lambda /. 2.) (add d2b_dt2 (mul (float kappa) db_dt))) in
  let denominator = add (float 2.) (mul (float lambda) (float kappa)) in
  
  div numerator denominator |> to_float0_exn

let two_trader_equilibrium lambda kappa gamma =
  let rec solve a b =
    let new_a t = best_response_strategy b lambda kappa t in
    let new_b t = best_response_strategy a (lambda /. gamma) kappa t in
    if abs_float (compute_cost a b lambda kappa gamma - compute_cost new_a new_b lambda kappa gamma) < 1e-6 then
      (new_a, new_b)
    else
      solve new_a new_b
  in
  solve risk_neutral_strategy risk_neutral_strategy

let multi_trader_symmetric_equilibrium n lambda kappa =
  let alpha = float n *. kappa /. (float n +. 2.) in
  fun t -> (exp (alpha *. t) -. 1.) /. (exp alpha -. 1.)

let n_trader_equilibrium n lambda kappa gamma =
  let rec solve strategies =
    let new_strategies = List.mapi (fun i s ->
      let others = List.filteri (fun j _ -> i <> j) strategies in
      let combined_others t = List.fold_left (fun acc s -> acc +. s t) 0. others in
      best_response_strategy combined_others (lambda /. float n) kappa
    ) strategies in
    if List.for_all2 (fun s1 s2 -> 
      abs_float (compute_cost s1 risk_neutral_strategy lambda kappa gamma - 
                 compute_cost s2 risk_neutral_strategy lambda kappa gamma) < 1e-6
    ) strategies new_strategies
    then new_strategies
    else solve new_strategies
  in
  solve (List.init n (fun _ -> risk_neutral_strategy))

let asymmetric_equilibrium scales lambda kappa gamma =
  let n = List.length scales in
  let rec solve strategies =
    let new_strategies = List.mapi (fun i s ->
      let others = List.filteri (fun j _ -> i <> j) strategies in
      let combined_others t = List.fold_left2 (fun acc scale s -> acc +. scale *. s t) 0. scales others in
      best_response_strategy combined_others (lambda /. List.nth scales i) kappa
    ) strategies in
    if List.for_all2 (fun s1 s2 -> 
      abs_float (compute_cost s1 risk_neutral_strategy lambda kappa gamma - 
                 compute_cost s2 risk_neutral_strategy lambda kappa gamma) < 1e-6
    ) strategies new_strategies
    then new_strategies
    else solve new_strategies
  in
  solve (List.init n (fun _ -> risk_neutral_strategy))

(* Risk Management *)
let calculate_risk measure returns =
  match measure with
  | ValueAtRisk confidence_level ->
      let sorted_returns = Array.copy returns |> Array.sort compare in
      let index = int_of_float (float (Array.length returns) *. (1. -. confidence_level)) in
      sorted_returns.(index)
  | ExpectedShortfall confidence_level ->
      let sorted_returns = Array.copy returns |> Array.sort compare in
      let cutoff_index = int_of_float (float (Array.length returns) *. (1. -. confidence_level)) in
      Array.sub sorted_returns 0 cutoff_index
      |> Array.fold_left (+.) 0.
      |> fun sum -> sum /. float cutoff_index
  | MaxDrawdown ->
      let max_so_far = ref (returns.(0)) in
      let max_drawdown = ref 0. in
      Array.iter (fun r ->
        max_so_far := max !max_so_far r;
        max_drawdown := max !max_drawdown (!max_so_far -. r)
      ) returns;
      !max_drawdown

let apply_risk_constraint strategy risk_measure max_risk =
  fun i t ->
    let original_allocation = strategy i t in
    let risk = calculate_risk risk_measure [|original_allocation|] in
    if risk > max_risk then
      original_allocation *. (max_risk /. risk)
    else
      original_allocation

(* Performance Metrics *)
let calculate_performance metric returns =
  let mean_return = Array.fold_left (+.) 0. returns /. float (Array.length returns) in
  let std_dev = 
    Array.fold_left (fun acc r -> acc +. (r -. mean_return) ** 2.) 0. returns
    |> fun sum -> sqrt (sum /. float (Array.length returns - 1))
  in
  match metric with
  | SharpeRatio -> mean_return /. std_dev
  | SortinoRatio ->
      let downside_returns = Array.filter (fun r -> r < 0.) returns in
      let downside_dev =
        Array.fold_left (fun acc r -> acc +. r ** 2.) 0. downside_returns
        |> fun sum -> sqrt (sum /. float (Array.length downside_returns))
      in
      mean_return /. downside_dev
  | InformationRatio benchmark ->
      let benchmark_returns = Array.init (Array.length returns) (fun i -> benchmark i (float i /. float (Array.length returns))) in
      let excess_returns = Array.map2 (-.) returns benchmark_returns in
      let mean_excess_return = Array.fold_left (+.) 0. excess_returns /. float (Array.length excess_returns) in
      let tracking_error =
        Array.fold_left (fun acc r -> acc +. (r -. mean_excess_return) ** 2.) 0. excess_returns
        |> fun sum -> sqrt (sum /. float (Array.length excess_returns - 1))
      in
      mean_excess_return /. tracking_error
  | CalmarRatio ->
      let max_drawdown = calculate_risk MaxDrawdown returns in
      mean_return /. max_drawdown

let rank_strategies strategies metric num_assets market_models lambda kappa gamma num_steps =
  let simulate_strategy strategy =
    let simulation = run_interactive_simulation num_assets 1000000. strategy market_models lambda kappa gamma num_steps in
    let returns = Array.init (num_steps - 1) (fun i ->
      let state1 = List.nth simulation i in
      let state2 = List.nth simulation (i + 1) in
      (state2.cash -. state1.cash) /. state1.cash
    ) in
    (strategy, calculate_performance metric returns)
  in
  List.map simulate_strategy strategies
  |> List.sort (fun (_, score1) (_, score2) -> compare score2 score1)

(* Probabilistic Strategy Selection *)
let create_lognormal_measure mu sigma =
  fun x ->
    let y = log x in
    exp (-. (y -. mu) ** 2. /. (2. *. sigma ** 2.)) /. (x *. sigma *. sqrt (2. *. Stdlib.pi))

let expected_cost s measure lambda kappa gamma =
  let integrand x =
    let scaled_s t = x *. (s t) in
    (compute_cost scaled_s risk_neutral_strategy lambda kappa gamma) *. (measure x)
  in
  (* Simple numerical integration *)
  let num_points = 1000 in
  let dx = 9.9 /. float num_points in
  let sum = ref 0. in
  for i = 0 to num_points - 1 do
    let x = 0.1 +. float i *. dx in
    sum := !sum +. integrand x *. dx
  done;
  !sum

(* Optimization Techniques *)
let particle_swarm_optimization objective num_particles num_iterations =
  let dim = 3 in  (* 3-dimensional strategy parameterization *)
  let particles = Array.init num_particles (fun _ -> Array.init dim (fun _ -> Random.float 2.0 -. 1.0)) in
  let velocities = Array.init num_particles (fun _ -> Array.init dim (fun _ -> 0.0)) in
  let best_positions = Array.copy particles in
  let best_scores = Array.map (fun p -> objective (Array.to_list p)) particles in
  let global_best_position = ref (Array.copy particles.(0)) in
  let global_best_score = ref best_scores.(0) in

  for _ = 1 to num_iterations do
    for i = 0 to num_particles - 1 do
      let score = objective (Array.to_list particles.(i)) in
      if score < best_scores.(i) then begin
        best_scores.(i) <- score;
        best_positions.(i) <- Array.copy particles.(i);
        if score < !global_best_score then begin
          global_best_score := score;
          global_best_position := Array.copy particles.(i);
        end
      end;

      for d = 0 to dim - 1 do
        let inertia = 0.5 in
        let cognitive = 1.5 in
        let social = 1.5 in
        velocities.(i).(d) <- inertia *. velocities.(i).(d) +.
                               cognitive *. Random.float 1.0 *. (best_positions.(i).(d) -. particles.(i).(d)) +.
                               social *. Random.float 1.0 *. (!global_best_position.(d) -. particles.(i).(d));
        particles.(i).(d) <- particles.(i).(d) +. velocities.(i).(d);
      done
    done
  done;
  
  (Array.to_list !global_best_position, !global_best_score)

let simulated_annealing objective initial_state initial_temp cooling_rate num_iterations =
  let current_state = ref initial_state in
  let current_energy = ref (objective !current_state) in
  let best_state = ref !current_state in
  let best_energy = ref !current_energy in
  let temp = ref initial_temp in

  for _ = 1 to num_iterations do
    let neighbor = List.map (fun x -> x +. (Random.float 0.2 -. 0.1)) !current_state in
    let neighbor_energy = objective neighbor in
    let delta_e = neighbor_energy -. !current_energy in

    if delta_e < 0.0 || Random.float 1.0 < exp (-. delta_e /. !temp) then begin
      current_state := neighbor;
      current_energy := neighbor_energy;
      if neighbor_energy < !best_energy then begin
        best_state := neighbor;
        best_energy := neighbor_energy;
      end
    end;

    temp := !temp *. cooling_rate;
  done;

  !best_state

let genetic_algorithm fitness_func pop_size gene_size num_generations mutation_rate crossover_rate =
  let create_individual () =
    let genes = Array.init gene_size (fun _ -> Random.float 2. -. 1.) in
    { genes; fitness = fitness_func (Array.to_list genes) }
  in

  let initial_population = Array.init pop_size create_individual in

  let select_parent population =
    let tournament_size = 5 in
    let tournament = Array.init tournament_size (fun _ -> population.(Random.int pop_size)) in
    Array.fold_left (fun best ind -> if ind.fitness > best.fitness then ind else best) tournament.(0) tournament
  in

  let crossover parent1 parent2 =
    Array.init gene_size (fun i ->
      if Random.float 1. < crossover_rate
      then parent1.genes.(i)
      else parent2.genes.(i)
    )
  in

  let mutate genes =
    Array.map (fun gene ->
      if Random.float 1. < mutation_rate
      then gene +. (Random.float 0.2 -. 0.1)
      else gene
    ) genes
  in

  let evolve_population population =
    Array.init pop_size (fun _ ->
      let parent1 = select_parent population in
      let parent2 = select_parent population in
      let child_genes = crossover parent1 parent2 |> mutate in
      { genes = child_genes; fitness = fitness_func (Array.to_list child_genes) }
    )
  in

  let rec evolve gen population =
    if gen = num_generations then
      Array.fold_left (fun best ind -> if ind.fitness > best.fitness then ind else best) population.(0) population
    else
      evolve (gen + 1) (evolve_population population)
  in

  (evolve 0 initial_population).genes |> Array.to_list

(* Portfolio Construction and Rebalancing *)
let construct_portfolio strategy num_assets =
  Array.init num_assets (fun i -> strategy i 0.)

let rebalance_portfolio portfolio strategy current_time prices =
  let new_weights = Array.mapi (fun i _ -> strategy i current_time) portfolio.weights in
  { weights = new_weights; last_rebalance = current_time }

let portfolio_optimization returns covariance risk_aversion =
  let num_assets = Array.length returns in
  let sigma = Tensor.of_float2 covariance in
  let mu = Tensor.of_float1 returns in

  let ones = Tensor.ones [num_assets] in
  let sigma_inv = Tensor.inverse sigma in
  
  let a = Tensor.(mm (mm (transpose ones) sigma_inv) ones |> to_float0_exn) in
  let b = Tensor.(mm (mm (transpose ones) sigma_inv) mu |> to_float0_exn) in
  let c = Tensor.(mm (mm (transpose mu) sigma_inv) mu |> to_float0_exn) in
  
  let lambda = (risk_aversion -. b) /. a in
  let weights = Tensor.(
    add (mul (float (1. /. risk_aversion)) (mm sigma_inv mu))
        (mul (float ((b -. risk_aversion) /. (risk_aversion *. a))) (mm sigma_inv ones))
  ) in
  
  Tensor.to_float1_exn weights

let black_litterman_optimization returns covariance views view_uncertainty tau risk_aversion constraints =
  let num_assets = Array.length returns in
  let prior_returns = Tensor.of_float1 returns in
  let sigma = Tensor.of_float2 covariance in
  
  let pi = Tensor.(mul (mul sigma prior_returns) (float risk_aversion)) in
  let omega = Tensor.(mul (mul (float tau) sigma) (of_float2 view_uncertainty)) in
  let p = Tensor.of_float2 views in
  let q = Tensor.(mv p prior_returns) in
  
  let term1 = Tensor.(add (mul (inverse sigma) (float risk_aversion)) (mm (transpose p) (inverse omega))) in
  let term2 = Tensor.(add (mv (inverse sigma) pi) (mv (mm (transpose p) (inverse omega)) q)) in
  let posterior_returns = Tensor.(mv (inverse term1) term2) in
  
  let weights = Tensor.(div posterior_returns (mul (float risk_aversion) (sum posterior_returns))) in
  
  let constrained_weights = List.fold_left (fun w constraint_ ->
    match constraint_ with
    | MaxWeight max_w -> Tensor.(min w (float max_w))
    | MinWeight min_w -> Tensor.(max w (float min_w))
    | SectorExposure (indices, min_exp, max_exp) ->
        let sector_exposure = Tensor.(sum (index w (Tensor.of_int1 indices))) in
        if Tensor.to_float0_exn sector_exposure < min_exp then
          Tensor.(add w (float ((min_exp -. Tensor.to_float0_exn sector_exposure) /. float (Array.length indices))))
        else if Tensor.to_float0_exn sector_exposure > max_exp then
          Tensor.(sub w (float ((Tensor.to_float0_exn sector_exposure -. max_exp) /. float (Array.length indices))))
        else w
  ) weights constraints in
  
  Tensor.to_float1_exn constrained_weights

let risk_parity_optimization covariance constraints =
  let num_assets = Array.length covariance in
  let sigma = Tensor.of_float2 covariance in
  
  let objective w =
    let risk_contributions = Tensor.(mul (mv sigma w) w) in
    let total_risk = Tensor.sum risk_contributions in
    Tensor.(sum (pow (div risk_contributions total_risk) (float 2.)))
  in
  
  let initial_weights = Tensor.(div (ones [num_assets]) (float num_assets)) in
  let optimizer = Optimizer.adam [initial_weights] ~learning_rate:0.01 in
  
  for _ = 1 to 1000 do
    let loss = objective initial_weights in
    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;
    
    initial_weights <- List.fold_left (fun w constraint_ ->
      match constraint_ with
      | MaxWeight max_w -> Tensor.(min w (float max_w))
      | MinWeight min_w -> Tensor.(max w (float min_w))
      | SectorExposure (indices, min_exp, max_exp) ->
          let sector_exposure = Tensor.(sum (index w (Tensor.of_int1 indices))) in
          if Tensor.to_float0_exn sector_exposure < min_exp then
            Tensor.(add w (float ((min_exp -. Tensor.to_float0_exn sector_exposure) /. float (Array.length indices))))
          else if Tensor.to_float0_exn sector_exposure > max_exp then
            Tensor.(sub w (float ((Tensor.to_float0_exn sector_exposure -. max_exp) /. float (Array.length indices))))
          else w
    ) initial_weights constraints
  done;
  
  Tensor.to_float1_exn initial_weights

let mean_variance_optimization returns covariance risk_aversion constraints =
  let num_assets = Array.length returns in
  let mu = Tensor.of_float1 returns in
  let sigma = Tensor.of_float2 covariance in
  
  let objective w =
    let return = Tensor.(sum (mul mu w)) in
    let risk = Tensor.(mm (mm (unsqueeze w 0) sigma) (unsqueeze w 1)) |> Tensor.squeeze |> Tensor.to_float0_exn in
    return -. risk_aversion *. risk
  in
  
  let initial_weights = Tensor.(div (ones [num_assets]) (float num_assets)) in
  let optimizer = Optimizer.adam [initial_weights] ~learning_rate:0.01 in
  
  for _ = 1 to 1000 do
    let loss = Tensor.neg (Tensor.of_float0 (objective initial_weights)) in
    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;
    
    initial_weights <- List.fold_left (fun w c -> if c w then w else Tensor.(abs w)) initial_weights constraints;
    initial_weights <- Tensor.(div w (sum w)); (* Normalize to sum to 1 *)
  done;
  
  Tensor.to_float1_exn initial_weights

let conditional_value_at_risk_optimization returns covariance confidence_level risk_aversion constraints =
  let num_assets = Array.length returns in
  let mu = Tensor.of_float1 returns in
  let sigma = Tensor.of_float2 covariance in
  
  let objective w =
    let portfolio_returns = Tensor.(mm (unsqueeze mu 0) (unsqueeze w 1)) |> Tensor.to_float0_exn in
    let portfolio_variance = Tensor.(mm (mm (unsqueeze w 0) sigma) (unsqueeze w 1)) |> Tensor.to_float0_exn in
    let portfolio_std = sqrt portfolio_variance in
    
    let var = Tensor.norm_cdf (Tensor.of_float0 ((1.0 -. confidence_level))) |> Tensor.to_float0_exn in
    let cvar = portfolio_returns -. (portfolio_std *. var /. (1.0 -. confidence_level)) in
    
    cvar -. risk_aversion *. portfolio_variance
  in
  
  let initial_weights = Tensor.(div (ones [num_assets]) (float num_assets)) in
  let optimizer = Optimizer.adam [initial_weights] ~learning_rate:0.01 in
  
  for _ = 1 to 1000 do
    let loss = Tensor.neg (Tensor.of_float0 (objective initial_weights)) in
    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;
    
    initial_weights <- List.fold_left (fun w c -> if c w then w else Tensor.(abs w)) initial_weights constraints;
    initial_weights <- Tensor.(div w (sum w)); (* Normalize to sum to 1 *)
  done;
  
  Tensor.to_float1_exn initial_weights

let robust_portfolio_optimization returns covariance uncertainty risk_aversion constraints =
  let num_assets = Array.length returns in
  let mu = Tensor.of_float1 returns in
  let sigma = Tensor.of_float2 covariance in
  
  let objective w =
    let portfolio_returns = Tensor.(mm (unsqueeze mu 0) (unsqueeze w 1)) |> Tensor.to_float0_exn in
    let portfolio_variance = Tensor.(mm (mm (unsqueeze w 0) sigma) (unsqueeze w 1)) |> Tensor.to_float0_exn in
    
    let worst_case_returns = portfolio_returns -. uncertainty *. sqrt portfolio_variance in
    
    worst_case_returns -. risk_aversion *. portfolio_variance
  in
  
  let initial_weights = Tensor.(div (ones [num_assets]) (float num_assets)) in
  let optimizer = Optimizer.adam [initial_weights] ~learning_rate:0.01 in
  
  for _ = 1 to 1000 do
    let loss = Tensor.neg (Tensor.of_float0 (objective initial_weights)) in
    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;
    
    initial_weights <- List.fold_left (fun w c -> if c w then w else Tensor.(abs w)) initial_weights constraints;
    initial_weights <- Tensor.(div w (sum w)); (* Normalize to sum to 1 *)
  done;
  
  Tensor.to_float1_exn initial_weights

let risk_budgeting_optimization covariance risk_budgets =
  let num_assets = Array.length risk_budgets in
  let sigma = Tensor.of_float2 covariance in
  
  let objective w =
    let risk_contributions = Tensor.(mul (mv sigma w) w) in
    let total_risk = Tensor.sum risk_contributions in
    Tensor.(sum (pow (sub (div risk_contributions total_risk) (of_float1 risk_budgets)) (float 2.)))
  in
  
  let initial_weights = Tensor.(div (ones [num_assets]) (float num_assets)) in
  let optimizer = Optimizer.adam [initial_weights] ~learning_rate:0.01 in
  
  for _ = 1 to 1000 do
    let loss = objective initial_weights in
    Optimizer.zero_grad optimizer;
    Tensor.backward loss;
    Optimizer.step optimizer;
    
    initial_weights <- Tensor.(abs initial_weights);
    initial_weights <- Tensor.(div initial_weights (sum initial_weights));
  done;
  
  Tensor.to_float1_exn initial_weights

(* Market Impact *)
let calculate_market_impact model volume price =
  match model with
  | Linear k -> 
      let impact = k *. volume in
      (impact, impact)
  | SquareRoot k -> 
      let impact = k *. sqrt (abs_float volume) *. (if volume >= 0. then 1. else -1.) in
      (impact, impact)
  | PowerLaw (k, alpha) -> 
      let impact = k *. (abs_float volume) ** alpha *. (if volume >= 0. then 1. else -1.) in
      (impact, impact)
  | TempermantonPermanent (k_temp, k_perm) -> 
      let temp_impact = k_temp *. sqrt (abs_float volume) *. (if volume >= 0. then 1. else -1.) in
      let perm_impact = k_perm *. volume in
      (temp_impact, perm_impact)

(* Multi-Factor Models *)
let create_factor name beta returns = { name; beta; returns }

let multi_factor_returns factors asset_betas =
  let num_periods = Array.length (List.hd factors).returns in
  Array.init num_periods (fun t ->
    List.fold_left2 (fun acc factor beta ->
      acc +. beta *. factor.beta *. factor.returns.(t)
    ) 0. factors asset_betas
  )

(* Advanced Analytics *)
let calculate_maximum_adverse_excursion returns =
  let max_drawdown = ref 0. in
  let peak = ref 1. in
  let current = ref 1. in
  Array.iter (fun r ->
    current := !current *. (1. +. r);
    if !current > !peak then peak := !current
    else
      let drawdown = (!peak -. !current) /. !peak in
      if drawdown > !max_drawdown then max_drawdown := drawdown
  ) returns;
  !max_drawdown

let calculate_maximum_favorable_excursion returns =
  let max_runup = ref 0. in
  let trough = ref 1. in
  let current = ref 1. in
  Array.iter (fun r ->
    current := !current *. (1. +. r);
    if !current < !trough then trough := !current
    else
      let runup = (!current -. !trough) /. !trough in
      if runup > !max_runup then max_runup := runup
  ) returns;
  !max_runup

let calculate_win_loss_ratio returns =
  let wins = ref 0 in
  let losses = ref 0 in
  Array.iter (fun r ->
    if r > 0. then incr wins
    else if r < 0. then incr losses
  ) returns;
  float !wins /. float !losses

let calculate_profit_factor returns =
  let gross_profit = ref 0. in
  let gross_loss = ref 0. in
  Array.iter (fun r ->
    if r > 0. then gross_profit := !gross_profit +. r
    else if r < 0. then gross_loss := !gross_loss -. r
  ) returns;
  !gross_profit /. !gross_loss

let calculate_beta strategy_returns market_returns =
  let strategy_var = Array.fold_left (fun acc r -> acc +. r ** 2.0) 0.0 strategy_returns /. float (Array.length strategy_returns) in
  let covariance = Array.fold_left2 (fun acc r1 r2 -> acc +. r1 *. r2) 0.0 strategy_returns market_returns /. float (Array.length strategy_returns) in
  covariance /. strategy_var

let calculate_alpha strategy_returns market_returns risk_free_rate =
  let beta = calculate_beta strategy_returns market_returns in
  let strategy_mean = Array.fold_left (+.) 0.0 strategy_returns /. float (Array.length strategy_returns) in
  let market_mean = Array.fold_left (+.) 0.0 market_returns /. float (Array.length market_returns) in
  strategy_mean -. (risk_free_rate +. beta *. (market_mean -. risk_free_rate))

let calculate_information_ratio strategy_returns benchmark_returns =
  let excess_returns = Array.map2 (-.) strategy_returns benchmark_returns in
  let mean_excess_return = Array.fold_left (+.) 0.0 excess_returns /. float (Array.length excess_returns) in
  let tracking_error = sqrt (Array.fold_left (fun acc r -> acc +. (r -. mean_excess_return) ** 2.0) 0.0 excess_returns /. float (Array.length excess_returns)) in
  mean_excess_return /. tracking_error

(* Time Series Analysis *)
let calculate_autocorrelation returns lag =
  let mean = Array.fold_left (+.) 0. returns /. float (Array.length returns) in
  let variance = Array.fold_left (fun acc r -> acc +. (r -. mean) ** 2.) 0. returns /. float (Array.length returns) in
  let n = Array.length returns - lag in
  let auto_cov = ref 0. in
  for i = 0 to n - 1 do
    auto_cov := !auto_cov +. (returns.(i) -. mean) *. (returns.(i + lag) -. mean)
  done;
  !auto_cov /. (float n *. variance)

let calculate_partial_autocorrelation returns max_lag =
  let n = Array.length returns in
  let mean = Array.fold_left (+.) 0. returns /. float n in
  let centered_returns = Array.map (fun r -> r -. mean) returns in
  
  let levinson_durbin phi k =
    let rec aux j acc =
      if j = 0 then acc
      else
        let num = centered_returns.(k) -. (Array.fold_left2 (fun s a b -> s +. a *. b) 0. (Array.sub phi 0 j) (Array.init j (fun i -> centered_returns.(k-i-1)))) in
        let den = 1. -. (Array.fold_left2 (fun s a b -> s +. a *. b) 0. (Array.sub phi 0 j) (Array.init j (fun i -> phi.(i)))) in
        let new_phi = Array.copy phi in
        new_phi.(j) <- num /. den;
        for i = 0 to j - 1 do
          new_phi.(i) <- phi.(i) -. new_phi.(j) *. phi.(j-i-1)
        done;
        aux (j-1) new_phi
    in
    aux k (Array.make (k+1) 0.)
  in
  
  Array.init (max_lag+1) (fun k -> 
    if k = 0 then 1.
    else (levinson_durbin (Array.make k 0.) k).(k-1)
  )

let perform_augmented_dickey_fuller_test returns =
  let n = Array.length returns in
  let y = Array.sub returns 1 (n-1) in
  let x = Array.sub returns 0 (n-1) in
  let dy = Array.map2 (fun a b -> a -. b) y x in
  
  let sum_x = Array.fold_left (+.) 0. x in
  let sum_y = Array.fold_left (+.) 0. y in
  let sum_xy = Array.fold_left2 (fun acc a b -> acc +. a *. b) 0. x y in
  let sum_xx = Array.fold_left (fun acc a -> acc +. a *. a) 0. x in
  
  let beta = (float (n-1) *. sum_xy -. sum_x *. sum_y) /. (float (n-1) *. sum_xx -. sum_x *. sum_x) in
  let alpha = (sum_y -. beta *. sum_x) /. float (n-1) in
  
  let residuals = Array.map2 (fun a b -> a -. (alpha +. beta *. b)) y x in
  let rss = Array.fold_left (fun acc r -> acc +. r *. r) 0. residuals in
  
  let se_beta = sqrt (rss /. (float (n-1) *. (sum_xx -. sum_x *. sum_x /. float (n-1)))) in
  let t_stat = (beta -. 1.) /. se_beta in
  
  (t_stat, beta)

(* Visualization *)
let plot_strategies strategies filename =
  let open Gnuplot in
  let gp = Gp.create () in
  Gp.plot_many gp ~range:(Range.({x1 = 0.; x2 = 1.; y1 = 0.; y2 = 1.}))
    (List.mapi (fun i (name, s) ->
      let xs = Array.init 100 (fun j -> float j /. 99.) in
      let ys = Array.map s xs in
      Series.lines_xy ~title:name xs ys
    ) strategies);
  Gp.output gp filename

let plot_drawdown_chart returns filename =
  let open Gnuplot in
  let gp = Gp.create () in
  let cumulative_returns = Array.make (Array.length returns) 1. in
  let drawdowns = Array.make (Array.length returns) 0. in
  let peak = ref 1. in
  for i = 1 to Array.length returns - 1 do
    cumulative_returns.(i) <- cumulative_returns.(i-1) *. (1. +. returns.(i));
    if cumulative_returns.(i) > !peak then peak := cumulative_returns.(i)
    else drawdowns.(i) <- (!peak -. cumulative_returns.(i)) /. !peak
  done;
  Gp.plot_many gp ~range:(Range.XY (0., float (Array.length returns), 0., 1.))
    [Series.lines_xy (Array.init (Array.length returns) float) drawdowns ~title:"Drawdown"];
  Gp.output gp filename

let plot_underwater_chart returns filename =
  let open Gnuplot in
  let gp = Gp.create () in
  let cumulative_returns = Array.make (Array.length returns) 1. in
  let underwater = Array.make (Array.length returns) 0. in
  let peak = ref 1. in
  for i = 1 to Array.length returns - 1 do
    cumulative_returns.(i) <- cumulative_returns.(i-1) *. (1. +. returns.(i));
    if cumulative_returns.(i) > !peak then peak := cumulative_returns.(i);
    underwater.(i) <- (cumulative_returns.(i) /. !peak) -. 1.
  done;
  Gp.plot_many gp ~range:(Range.XY (0., float (Array.length returns), -1., 0.))
    [Series.lines_xy (Array.init (Array.length returns) float) underwater ~title:"Underwater"];
  Gp.output gp filename

let plot_rolling_sharpe_ratio returns window filename =
  let open Gnuplot in
  let gp = Gp.create () in
  let sharpe_ratios = Array.make (Array.length returns - window + 1) 0. in
  for i = 0 to Array.length returns - window do
    let window_returns = Array.sub returns i window in
    let mean = Array.fold_left (+.) 0. window_returns /. float window in
    let std_dev = sqrt (Array.fold_left (fun acc r -> acc +. (r -. mean) ** 2.) 0. window_returns /. float window) in
    sharpe_ratios.(i) <- mean /. std_dev *. sqrt 252.
  done;
  Gp.plot_many gp ~range:(Range.XY (0., float (Array.length sharpe_ratios), 0., 5.))
    [Series.lines_xy (Array.init (Array.length sharpe_ratios) float) sharpe_ratios ~title:"Rolling Sharpe Ratio"];
  Gp.output gp filename

let plot_rolling_sortino_ratio returns window filename =
  let open Gnuplot in
  let gp = Gp.create () in
  let sortino_ratios = Array.make (Array.length returns - window + 1) 0. in
  for i = 0 to Array.length returns - window do
    let window_returns = Array.sub returns i window in
    let mean = Array.fold_left (+.) 0. window_returns /. float window in
    let downside_deviation = sqrt (Array.fold_left (fun acc r -> acc +. (min r 0.) ** 2.) 0. window_returns /. float window) in
    sortino_ratios.(i) <- mean /. downside_deviation *. sqrt 252.
  done;
  Gp.plot_many gp ~range:(Range.XY (0., float (Array.length sortino_ratios), 0., 5.))
    [Series.lines_xy (Array.init (Array.length sortino_ratios) float) sortino_ratios ~title:"Rolling Sortino Ratio"];
  Gp.output gp filename

let plot_rolling_beta strategy_returns market_returns window filename =
  let open Gnuplot in
  let gp = Gp.create () in
  let betas = Array.make (Array.length strategy_returns - window + 1) 0. in
  for i = 0 to Array.length strategy_returns - window do
    let strategy_window = Array.sub strategy_returns i window in
    let market_window = Array.sub market_returns i window in
    betas.(i) <- calculate_beta strategy_window market_window
  done;
  Gp.plot_many gp ~range:(Range.XY (0., float (Array.length betas), -1., 2.))
    [Series.lines_xy (Array.init (Array.length betas) float) betas ~title:"Rolling Beta"];
  Gp.output gp filename

let plot_efficient_frontier strategies num_assets market_models lambda kappa gamma num_steps filename =
  let open Gnuplot in
  let gp = Gp.create () in
  
  let points = List.map (fun strategy ->
    let simulation = run_interactive_simulation num_assets 1000000. strategy market_models lambda kappa gamma num_steps in
    let returns = Array.init (num_steps - 1) (fun i ->
      let state1 = List.nth simulation i in
      let state2 = List.nth simulation (i + 1) in
      (state2.cash -. state1.cash) /. state1.cash
    ) in
    let risk = calculate_risk (ValueAtRisk 0.95) returns in
    let return = calculate_performance SharpeRatio returns in
    (risk, return)
  ) strategies in
  
  Gp.plot_many gp ~range:(Range.XY (0., 0.1, 0., 2.))
    [Series.points_xy (Array.of_list (List.map fst points)) (Array.of_list (List.map snd points)) ~title:"Efficient Frontier"];
  
  Gp.output gp filename

let plot_strategy_comparison strategies num_assets market_models lambda kappa gamma num_steps filename =
  let open Gnuplot in
  let gp = Gp.create () in
  
  let simulations = List.map (fun strategy ->
    run_interactive_simulation num_assets 1000000. strategy market_models lambda kappa gamma num_steps
  ) strategies in
  
  let plot_data = List.mapi (fun i simulation ->
    let times = List.map (fun state -> state.time) simulation in
    let values = List.map (fun state -> 
      state.cash +. Array.fold_left2 (fun acc pos price -> acc +. pos *. price) 0. state.positions state.prices
    ) simulation in
    Series.lines_xy (Array.of_list times) (Array.of_list values) ~title:(Printf.sprintf "Strategy %d" (i + 1))
  ) simulations in
  
  Gp.plot_many gp ~range:(Range.XY (0., 1., 900000., 1100000.)) plot_data;
  
  Gp.output gp filename

(* Strategy Combination and Ensemble Methods *)
let combine_strategies strategies method_ =
  match method_ with
  | EqualWeight ->
      let num_strategies = List.length strategies in
      let weight = 1. /. float num_strategies in
      fun i t -> List.fold_left (fun acc s -> acc +. weight *. s i t) 0. strategies
  | InverseVolatility ->
      let calculate_volatility s =
        let returns = Array.init 252 (fun i -> s 0 (float i /. 252.) -. s 0 ((float i -. 1.) /. 252.)) in
        let mean = Array.fold_left (+.) 0. returns /. 252. in
        sqrt (Array.fold_left (fun acc r -> acc +. (r -. mean) ** 2.) 0. returns /. 251.)
      in
      let volatilities = List.map calculate_volatility strategies in
      let total_inverse_vol = List.fold_left (+.) 0. (List.map (fun v -> 1. /. v) volatilities) in
      let weights = List.map (fun v -> 1. /. (v *. total_inverse_vol)) volatilities in
      fun i t -> List.fold_left2 (fun acc s w -> acc +. w *. s i t) 0. strategies weights
  | OptimalF ->
      let calculate_optimal_f s =
        let returns = Array.init 252 (fun i -> s 0 (float i /. 252.) -. s 0 ((float i -. 1.) /. 252.)) in
        let wins = Array.fold_left (fun acc r -> if r > 0. then acc +. 1. else acc) 0. returns in
        let win_ratio = wins /. 252. in
        win_ratio /. (1. -. win_ratio)
      in
      let f_values = List.map calculate_optimal_f strategies in
      let total_f = List.fold_left (+.) 0. f_values in
      let weights = List.map (fun f -> f /. total_f) f_values in
      fun i t -> List.fold_left2 (fun acc s w -> acc +. w *. s i t) 0. strategies weights
  | KellyWeights ->
      let calculate_kelly s =
        let returns = Array.init 252 (fun i -> s 0 (float i /. 252.) -. s 0 ((float i -. 1.) /. 252.)) in
        let mean = Array.fold_left (+.) 0. returns /. 252. in
        let variance = Array.fold_left (fun acc r -> acc +. (r -. mean) ** 2.) 0. returns /. 251. in
        mean /. variance
      in
      let kelly_fractions = List.map calculate_kelly strategies in
      let total_kelly = List.fold_left (+.) 0. kelly_fractions in
      let weights = List.map (fun k -> k /. total_kelly) kelly_fractions in
      fun i t -> List.fold_left2 (fun acc s w -> acc +. w *. s i t) 0. strategies weights

let ensemble_strategy strategies weights =
  fun i t ->
    List.fold_left2 (fun acc strategy weight ->
      acc +. weight *. strategy i t
    ) 0. strategies weights

let adaptive_strategy strategies weight_func =
  fun i t ->
    let weights = weight_func t in
    List.fold_left2 (fun acc strategy weight ->
      acc +. weight *. strategy i t
    ) 0. strategies weights

(* Adaptive Strategy Optimization *)
let detect_market_regime returns window =
  let recent_returns = Array.sub returns (Array.length returns - window) window in
  let mean = Array.fold_left (+.) 0. recent_returns /. float window in
  let std_dev = sqrt (Array.fold_left (fun acc r -> acc +. (r -. mean) ** 2.) 0. recent_returns /. float window) in
  let annualized_return = mean *. sqrt 252. in
  let annualized_vol = std_dev *. sqrt 252. in
  
  if annualized_return > 0.15 && annualized_vol < 0.2 then Bull
  else if annualized_return < -0.15 && annualized_vol < 0.2 then Bear
  else if abs_float annualized_return < 0.05 && annualized_vol < 0.1 then Sideways
  else Volatile

let optimize_strategy_for_regime original_strategy regime =
  match regime with
  | Bull -> fun i t -> min 1.5 (1.2 *. original_strategy i t)
  | Bear -> fun i t -> max 0.5 (0.8 *. original_strategy i t)
  | Sideways -> original_strategy
  | Volatile -> fun i t -> 0.7 *. original_strategy i t

(* Transaction Cost Optimization *)
let optimize_execution_with_transaction_costs strategy impact_model lambda num_steps =
  let original_trades i t = strategy i t in
  
  let optimize_trades trades =
    let n = Array.length trades in
    let optimized_trades = Array.copy trades in
    let total_trade = Array.fold_left (+.) 0. trades in
    
    let objective trade_sizes =
      let cumulative_trades = Array.make (n+1) 0. in
      for i = 1 to n do
        cumulative_trades.(i) <- cumulative_trades.(i-1) +. trade_sizes.(i-1)
      done;
      
      let total_cost = ref 0. in
      for i = 0 to n-1 do
        let (temp_impact, perm_impact) = calculate_market_impact impact_model trade_sizes.(i) 1.0 in
        total_cost := !total_cost +. temp_impact *. abs_float trade_sizes.(i) +. 
                      perm_impact *. (total_trade -. cumulative_trades.(i))
      done;
      !total_cost +. lambda *. Array.fold_left (fun acc t -> acc +. t ** 2.) 0. trade_sizes
    in
    
    let optimizer = Optimizer.adam [Tensor.of_float1 optimized_trades] ~learning_rate:0.01 in
    
    for _ = 1 to 100 do
      let loss = Tensor.of_float0 (objective optimized_trades) in
      Optimizer.zero_grad optimizer;
      Tensor.backward loss;
      Optimizer.step optimizer
    done;
    
    Tensor.to_float1_exn (List.hd (Optimizer.parameters optimizer))
  in
  
  fun i t ->
    let t_index = int_of_float (t *. float num_steps) in
    let trades = Array.init num_steps (fun j -> original_trades i (float j /. float num_steps)) in
    let optimized_trades = optimize_trades trades in
    optimized_trades.(t_index)

(* Backtesting *)
let backtest_strategy strategy num_assets market_models cost_model lambda kappa gamma num_steps =
  let initial_capital = 1_000_000.0 in
  let positions = Array.make num_assets 0.0 in
  let cash = ref initial_capital in
  let pnl = ref 0.0 in
  let total_cost = ref 0.0 in
  let returns = Array.make num_steps 0.0 in
  let prices = Array.init num_assets (fun i -> (market_models i).price_process 100.0 0.0) in
  
  for t = 1 to num_steps do
    let portfolio_value = !cash +. Array.fold_left2 (fun acc pos price -> acc +. pos *. price) 0.0 positions prices in
    
    (* Calculate target positions *)
    let target_positions = Array.mapi (fun i _ -> strategy i (float t /. float num_steps) *. portfolio_value /. prices.(i)) positions in
    
    (* Execute trades *)
    Array.iteri (fun i target_pos ->
      let trade_volume = target_pos -. positions.(i) in
      let (temp_impact, perm_impact) = calculate_market_impact cost_model trade_volume prices.(i) in
      let trade_cost = temp_impact *. abs_float trade_volume in
      cash := !cash -. trade_volume *. prices.(i) -. trade_cost;
      total_cost := !total_cost +. trade_cost;
      positions.(i) <- target_pos;
      prices.(i) <- prices.(i) *. (1. +. perm_impact);
    ) target_positions;
    
    (* Update prices *)
    Array.iteri (fun i _ -> 
      prices.(i) <- (market_models i).price_process prices.(i) (float t /. float num_steps)
    ) prices;
    
    (* Calculate returns *)
    let new_portfolio_value = !cash +. Array.fold_left2 (fun acc pos price -> acc +. pos *. price) 0.0 positions prices in
    returns.(t-1) <- (new_portfolio_value -. portfolio_value) /. portfolio_value;
    pnl := !pnl +. new_portfolio_value -. portfolio_value
  done;
  
  let sharpe_ratio = 
    let mean_return = Array.fold_left (+.) 0.0 returns /. float num_steps in
    let std_dev = sqrt (Array.fold_left (fun acc r -> acc +. (r -. mean_return) ** 2.0) 0.0 returns /. float num_steps) in
    mean_return /. std_dev *. sqrt (252.0) (* Annualized Sharpe Ratio *)
  in
  
  let max_drawdown =
    let rec calculate_max_drawdown peak mdd i =
      if i = num_steps then mdd
      else
        let current_value = initial_capital *. (1.0 +. Array.fold_left (+.) 0.0 (Array.sub returns 0 i)) in
        let new_peak = max peak current_value in
        let new_mdd = max mdd ((new_peak -. current_value) /. new_peak) in
        calculate_max_drawdown new_peak new_mdd (i + 1)
    in
    calculate_max_drawdown initial_capital 0.0 0
  in
  
  { returns; sharpe_ratio; max_drawdown; total_pnl = !pnl; total_cost = !total_cost }

let advanced_backtest_strategy strategy num_assets market_models cost_model lambda kappa gamma num_steps =
  let base_result = backtest_strategy strategy num_assets market_models cost_model lambda kappa gamma num_steps in
  
  let var = calculate_risk_metric (ValueAtRisk (0.95, Historical)) base_result.returns in
  let cvar = calculate_risk_metric (ConditionalVaR (0.95, Historical)) base_result.returns in
  
  let calmar_ratio = 
    let annualized_return = (Array.fold_left (+.) 0. base_result.returns /. float (Array.length base_result.returns)) *. 252. in
    annualized_return /. base_result.max_drawdown
  in
  
  let threshold = 0. in
  let omega_ratio =
    let gains = Array.fold_left (fun acc r -> if r > threshold then acc +. r -. threshold else acc) 0. base_result.returns in
    let losses = Array.fold_left (fun acc r -> if r < threshold then acc +. threshold -. r else acc) 0. base_result.returns in
    gains /. losses
  in
  
  let sortino_ratio =
    let mean_return = Array.fold_left (+.) 0. base_result.returns /. float (Array.length base_result.returns) in
    let downside_deviation = calculate_risk_metric (DownsideDeviation 0.) base_result.returns in
    mean_return /. downside_deviation
  in
  
  { base_result; var; cvar; calmar_ratio; omega_ratio; sortino_ratio }

(* Performance Reporting *)
let generate_performance_report result filename =
  let oc = open_out filename in
  Printf.fprintf oc "Performance Report\n";
  Printf.fprintf oc "-------------------\n";
  Printf.fprintf oc "Sharpe Ratio: %.4f\n" result.sharpe_ratio;
  Printf.fprintf oc "Max Drawdown: %.2f%%\n" (result.max_drawdown *. 100.0);
  Printf.fprintf oc "Total PnL: $%.2f\n" result.total_pnl;
  Printf.fprintf oc "Total Cost: $%.2f\n" result.total_cost;
  Printf.fprintf oc "Annualized Return: %.2f%%\n" ((Array.fold_left (+.) 0.0 result.returns /. float (Array.length result.returns)) *. 252.0 *. 100.0);
  close_out oc

let compare_strategies_report results filename =
  let oc = open_out filename in
  Printf.fprintf oc "Strategy Comparison Report\n";
  Printf.fprintf oc "---------------------------\n";
  List.iter (fun (name, result) ->
    Printf.fprintf oc "\nStrategy: %s\n" name;
    Printf.fprintf oc "Sharpe Ratio: %.4f\n" result.sharpe_ratio;
    Printf.fprintf oc "Max Drawdown: %.2f%%\n" (result.max_drawdown *. 100.0);
    Printf.fprintf oc "Total PnL: $%.2f\n" result.total_pnl;
    Printf.fprintf oc "Total Cost: $%.2f\n" result.total_cost;
    Printf.fprintf oc "Annualized Return: %.2f%%\n" ((Array.fold_left (+.) 0.0 result.returns /. float (Array.length result.returns)) *. 252.0 *. 100.0)
  ) results;
  close_out oc

(* Utility Functions *)
let discretize_strategy s n =
  Array.init n (fun i -> s (float i /. float (n - 1)))

let interpolate_strategy arr =
  let n = Array.length arr in
  fun t ->
    let idx = int_of_float (t *. float (n - 1)) in
    if idx >= n - 1 then arr.(n - 1)
    else
      let t0 = float idx /. float (n - 1) in
      let t1 = float (idx + 1) /. float (n - 1) in
      let y0 = arr.(idx) in
      let y1 = arr.(idx + 1) in
      y0 +. (y1 -. y0) *. (t -. t0) /. (t1 -. t0)

let strategy_to_tensor s n =
  Tensor.of_float1 (Array.init n (fun i -> s (float i /. float (n - 1))))

let tensor_to_strategy t =
  let arr = Tensor.to_float1_exn t in
  let n = Array.length arr in
  fun x ->
    let idx = int_of_float (x *. float (n - 1)) in
    if idx >= n - 1 then arr.(n - 1)
    else
      let x0 = float idx /. float (n - 1) in
      let x1 = float (idx + 1) /. float (n - 1) in
      let y0 = arr.(idx) in
      let y1 = arr.(idx + 1) in
      y0 +. (y1 -. y0) *. (x -. x0) /. (x1 -. x0)

let strategy_metrics s lambda kappa gamma =
  let cost = compute_cost s risk_neutral_strategy lambda kappa gamma in
  let (d_lambda, d_kappa) = strategy_sensitivity s lambda kappa gamma in
  let max_deviation = 
    let max_dev = ref 0. in
    for i = 0 to 99 do
      let t = float i /. 99. in
      max_dev := max !max_dev (abs_float (s t -. t))
    done;
    !max_dev
  in
  (cost, max_deviation, (d_lambda +. d_kappa) /. 2.)

let strategy_moments s =
  let t_values = List.init 1001 (fun i -> float_of_int i /. 1000.0) in
  let values = List.map s t_values in
  let mean = List.fold_left (+.) 0.0 values /. 1001.0 in
  let central_moments = List.map (fun v -> (v -. mean) ** 4.0) values in
  let variance = List.fold_left (+.) 0.0 central_moments /. 1001.0 in
  let skewness = List.fold_left (fun acc v -> acc +. (v -. mean) ** 3.0) 0.0 values /. (1001.0 *. variance ** 1.5) in
  let kurtosis = List.fold_left (fun acc v -> acc +. (v -. mean) ** 4.0) 0.0 values /. (1001.0 *. variance ** 2.0) in
  (mean, variance, skewness, kurtosis)

let strategy_entropy s =
  let t_values = List.init 1001 (fun i -> float_of_int i /. 1000.0) in
  let values = List.map s t_values in
  let total = List.fold_left (+.) 0.0 values in
  let probabilities = List.map (fun v -> v /. total) values in
  -. List.fold_left (fun acc p -> if p > 0.0 then acc +. p *. log p else acc) 0.0 probabilities

let strategy_complexity s =
  let t_values = List.init 1000 (fun i -> float_of_int i /. 999.0) in
  let values = List.map s t_values in
  let diffs = List.map2 (fun a b -> abs_float (b -. a)) values (List.tl values) in
  List.fold_left (+.) 0.0 diffs

let perform_sensitivity_analysis strategy num_assets market_models lambda kappa gamma num_steps =
  let base_simulation = run_interactive_simulation num_assets 1000000. strategy market_models lambda kappa gamma num_steps in
  let base_value = (List.hd (List.rev base_simulation)).cash in
  
  let perturb_param param delta =
    match param with
    | "lambda" -> run_interactive_simulation num_assets 1000000. strategy market_models (lambda +. delta) kappa gamma num_steps
    | "kappa" -> run_interactive_simulation num_assets 1000000. strategy market_models lambda (kappa +. delta) gamma num_steps
    | "gamma" -> run_interactive_simulation num_assets 1000000. strategy market_models lambda kappa (gamma +. delta) num_steps
    | _ -> failwith "Unknown parameter"
  in
  
  let calculate_sensitivity param =
    let delta = 0.01 in
    let perturbed_simulation = perturb_param param delta in
    let perturbed_value = (List.hd (List.rev perturbed_simulation)).cash in
    (perturbed_value -. base_value) /. (delta *. base_value)
  in
  
  [("lambda", calculate_sensitivity "lambda");
   ("kappa", calculate_sensitivity "kappa");
   ("gamma", calculate_sensitivity "gamma")]

let generate_monte_carlo_paths model num_paths num_steps =
  Array.init num_paths (fun _ ->
    Array.init num_steps (fun i ->
      let t = float i /. float num_steps in
      model.price_process 100. t  (* Iniital price of 100 *)
    )
  )

(* Machine Learning-based Strategy Improvement *)
module MLStrategyImprovement = struct
  type model = {
    network: Layer.t;
    optimizer: Optimizer.t;
  }

  let create_model input_size hidden_sizes output_size =
    let layers = 
      (Layer.linear input_size (List.hd hidden_sizes) :: 
       List.map2 (fun in_size out_size -> Layer.linear in_size out_size) hidden_sizes (List.tl hidden_sizes @ [output_size]))
      |> List.intersperse (Layer.relu ())
    in
    let network = Layer.sequential layers in
    let optimizer = Optimizer.adam (Layer.parameters network) ~learning_rate:0.001 in
    { network; optimizer }

  let train_model model features targets num_epochs =
    let x = Tensor.of_float2 features in
    let y = Tensor.of_float1 targets in
    
    for _ = 1 to num_epochs do
      let predicted = Layer.forward model.network x in
      let loss = Tensor.mse_loss predicted y in
      Optimizer.zero_grad model.optimizer;
      Tensor.backward loss;
      Optimizer.step model.optimizer
    done

  let predict model features =
    let x = Tensor.of_float1 features in
    let predicted = Layer.forward model.network (Tensor.unsqueeze x 0) in
    Tensor.to_float0_exn (Tensor.squeeze predicted)

  let improve_strategy original_strategy feature_extractor model =
    fun i t ->
      let features = feature_extractor i in
      let original_allocation = original_strategy i t in
      let adjustment = predict model features in
      original_allocation +. adjustment
end

(* Reinforcement Learning Optimization *)
module DDPG = struct
  type actor = {
    model: Layer.t;
    target_model: Layer.t;
    optimizer: Optimizer.t;
  }

  type critic = {
    model: Layer.t;
    target_model: Layer.t;
    optimizer: Optimizer.t;
  }

  type experience = {
    state: Tensor.t;
    action: Tensor.t;
    reward: float;
    next_state: Tensor.t;
    done_: bool;
  }

  type replay_buffer = experience Queue.t

  let create_actor input_dim output_dim =
    let model = Layer.sequential [
      Layer.linear input_dim 64;
      Layer.relu;
      Layer.linear 64 32;
      Layer.relu;
      Layer.linear 32 output_dim;
      Layer.tanh;
    ] in
    let target_model = Layer.sequential [
      Layer.linear input_dim 64;
      Layer.relu;
      Layer.linear 64 32;
      Layer.relu;
      Layer.linear 32 output_dim;
      Layer.tanh;
    ] in
    let optimizer = Optimizer.adam (Layer.parameters model) ~learning_rate:1e-3 in
    { model; target_model; optimizer }

  let create_critic input_dim action_dim =
    let model = Layer.sequential [
      Layer.linear (input_dim + action_dim) 64;
      Layer.relu;
      Layer.linear 64 32;
      Layer.relu;
      Layer.linear 32 1;
    ] in
    let target_model = Layer.sequential [
      Layer.linear (input_dim + action_dim) 64;
      Layer.relu;
      Layer.linear 64 32;
      Layer.relu;
      Layer.linear 32 1;
    ] in
    let optimizer = Optimizer.adam (Layer.parameters model) ~learning_rate:1e-3 in
    { model; target_model; optimizer }

  let create_replay_buffer capacity = Queue.create ()

  let train_ddpg actor critic replay_buffer market_models num_assets num_episodes batch_size gamma tau =
    let update_target_model target_model model =
      List.iter2 (fun target_param param ->
        Tensor.(copy_ (add (mul target_param (float (1. -. tau))) (mul param (float tau))) target_param)
      ) (Layer.parameters target_model) (Layer.parameters model)
    in

    let select_action actor state =
      let action = Layer.forward actor.model state in
      Tensor.(add action (randn ~shape:(shape action) ~device:(device action)))
    in

    let train_models actor critic experiences =
      let states = Tensor.stack (List.map (fun e -> e.state) experiences) in
      let actions = Tensor.stack (List.map (fun e -> e.action) experiences) in
      let rewards = Tensor.of_float1 (Array.of_list (List.map (fun e -> e.reward) experiences)) in
      let next_states = Tensor.stack (List.map (fun e -> e.next_state) experiences) in
      let dones = Tensor.of_float1 (Array.of_list (List.map (fun e -> if e.done_ then 1. else 0.) experiences)) in

      let target_actions = Layer.forward actor.target_model next_states in
      let target_q_values = Layer.forward critic.target_model Tensor.(cat[next_states; target_actions] ~dim:1) in
      let target_q = Tensor.(add rewards (mul (float gamma) (mul target_q_values (sub (float 1.) dones)))) in

      (* Update Critic *)
      let current_q = Layer.forward critic.model Tensor.(cat [states; actions] ~dim:1) in
      let critic_loss = Tensor.mse_loss current_q target_q in
      Optimizer.zero_grad critic.optimizer;
      Tensor.backward critic_loss;
      Optimizer.step critic.optimizer;

      (* Update Actor *)
      let actor_actions = Layer.forward actor.model states in
      let actor_loss = Tensor.neg (Tensor.mean (Layer.forward critic.model Tensor.(cat [states; actor_actions] ~dim:1))) in
      Optimizer.zero_grad actor.optimizer;
      Tensor.backward actor_loss;
      Optimizer.step actor.optimizer;

      update_target_model actor.target_model actor.model;
      update_target_model critic.target_model critic.model
    in

    for episode = 1 to num_episodes do
      let initial_state = Tensor.rand [num_assets * 2] in (* state: prices and positions *)
      let rec simulate state total_reward step =
        if step = 100 then total_reward
        else
          let action = select_action actor state in
          let next_state, reward, done_ =
            Tensor.rand [num_assets * 2], Random.float 2. -. 1., false
          in
          Queue.add { state; action; reward; next_state; done_ } replay_buffer;
          if Queue.length replay_buffer > batch_size then begin
            let batch = List.init batch_size (fun _ -> Queue.take replay_buffer) in
            train_models actor critic batch
          end;
          simulate next_state (total_reward +. reward) (step + 1)
      in
      let episode_reward = simulate initial_state 0. 0 in
      Printf.printf "Episode %d: Total Reward = %f\n" episode episode_reward
    done;
    actor
end

(* Simulation *)
let initialize_simulation num_assets initial_cash =
  {
    time = 0.;
    prices = Array.init num_assets (fun _ -> 100.);  (* Starting price of 100 for all assets *)
    positions = Array.make num_assets 0.;
    cash = initial_cash;
  }

let simulation_step state strategy market_models lambda kappa gamma =
  let num_assets = Array.length state.prices in
  let dt = 0.01 in  (* Small time step *)
  let new_prices = Array.mapi (fun i p -> 
    let model = market_models i in
    model.price_process p dt
  ) state.prices in
  let new_positions = Array.mapi (fun i pos ->
    let target_pos = strategy i state.time in
    let trade = (target_pos -. pos) *. dt in
    pos +. trade
  ) state.positions in
  let cash_change = ref 0. in
  for i = 0 to num_assets - 1 do
    let price_impact = lambda *. (new_positions.(i) -. state.positions.(i)) in
    cash_change := !cash_change -. new_prices.(i) *. (new_positions.(i) -. state.positions.(i)) *. (1. +. price_impact)
  done;
  let new_cash = state.cash +. !cash_change in
  let new_state = {
    time = state.time +. dt;
    prices = new_prices;
    positions = new_positions;
    cash = new_cash;
  } in
  let pnl = new_cash +. Array.fold_left2 (fun acc pos price -> acc +. pos *. price) 0. new_positions new_prices
            -. (state.cash +. Array.fold_left2 (fun acc pos price -> acc +. pos *. price) 0. state.positions state.prices) in
  (new_state, pnl)

let run_interactive_simulation num_assets initial_cash strategy market_models lambda kappa gamma num_steps =
  let rec simulate state steps acc =
    if steps = 0 then List.rev acc
    else
      let (new_state, pnl) = simulation_step state strategy market_models lambda kappa gamma in
      simulate new_state (steps - 1) (new_state :: acc)
  in
  let initial_state = initialize_simulation num_assets initial_cash in
  simulate initial_state num_steps [initial_state]