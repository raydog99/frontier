open Torch

module SDE = struct
  type model = GBM | IGBM | CIR

  type t = {
    model: model;
    r: float;
    init_price: float;
    time_horizon: float;
  }

  let drift model s = match model with
    | GBM -> 0.05 *. s 
    | IGBM -> 2. *. (100. -. s)
    | CIR -> 2. *. (100. -. s)

  let volatility model s = match model with
    | GBM -> 0.2 *. s
    | IGBM -> 0.2 *. s 
    | CIR -> 0.2 *. sqrt s

  let volatility_derivative model s = match model with
    | GBM -> 0.2
    | IGBM -> 0.2
    | CIR -> 0.1 /. sqrt s
end

module Discretization = struct
  type scheme = Euler | Milstein

  let simulate_path sde scheme steps dt noise =
    let path = zeros [steps + 1] in
    index_put_ path [Some 0] (float_vec [sde.SDE.init_price]);
    
    let rec step_path i prev_price =
      if i >= steps then path
      else
        let w = get noise i |> float_value in
        let s = float_value prev_price in
        let next_price = match scheme with
        | Euler ->
            s +. SDE.drift sde.SDE.model s *. dt +. 
            SDE.volatility sde.SDE.model s *. w
        | Milstein ->
            let vol = SDE.volatility sde.SDE.model s in
            let vol_deriv = SDE.volatility_derivative sde.SDE.model s in
            s +. SDE.drift sde.SDE.model s *. dt +.
            vol *. w +. 0.5 *. vol *. vol_deriv *. (w *. w -. dt)
        in
        index_put_ path [Some (i + 1)] (float_vec [next_price]);
        step_path (i + 1) (float_vec [next_price])
    in
    step_path 0 (float_vec [sde.SDE.init_price])
end

module Estimator = struct
  type t = {
    sde: SDE.t;
    payoff: Payoff.t;
    scheme: Discretization.scheme;
    level: int;
    base: int;
  }

  type level_data = {
    estimator: t;
    variance: float;
    correlation: float;
    cost: float;
    samples: int;
  }

  let create sde payoff scheme level base = {
    sde;
    payoff;
    scheme;
    level;
    base;
  }

  let steps t = t.base * (1 lsl t.level)

  let dt t = t.sde.time_horizon /. float_of_int (steps t)

  let single_path t =
    let n_steps = steps t in
    let dt = dt t in
    let noise = Tensor.randn [n_steps] in
    let path = Discretization.simulate_path t.sde t.scheme n_steps dt noise in
    Payoff.compute t.payoff t.sde path

  let estimate t n_samples =
    let rec loop i acc =
      if i >= n_samples then acc /. float_of_int n_samples
      else loop (i + 1) (acc +. single_path t)
    in
    loop 0 0.
end

module MLMC = struct
  type t = {
    levels: Estimator.level_data array;
    target_variance: float;
  }

  let create sde scheme payoff target_var =
    let base_estimator = Estimator.create sde payoff scheme 0 32 in
    let max_level = 10 in
    
    let levels = Array.init (max_level + 1) (fun l ->
      let est = Estimator.create sde payoff scheme l 32 in
      let initial_samples = 100 in
      let samples = Array.init initial_samples (fun _ -> 
        Estimator.single_path est) in
      {
        Estimator.estimator = est;
        variance = Stats.variance samples;
        correlation = if l = 0 then 1.0 else
          Stats.correlation samples 
            (Array.init initial_samples (fun _ -> 
              Estimator.single_path base_estimator));
        cost = float_of_int (Estimator.steps est);
        samples = initial_samples;
      }
    ) in
    { levels; target_variance = target_var }

  let compute_optimal_samples t =
    let total_work = ref 0. in
    Array.iter (fun l -> 
      total_work := !total_work +. 
        sqrt (l.variance *. l.cost)
    ) t.levels;
    
    Array.map (fun l ->
      let n_opt = ceil (!total_work *. 
        sqrt (l.variance /. l.cost) /. t.target_variance) in
      int_of_float n_opt
    ) t.levels

  let estimate t =
    let optimal_samples = compute_optimal_samples t in
    Array.fold_left2 (fun acc level n_samples ->
      acc +. Estimator.estimate level.estimator n_samples
    ) 0. t.levels optimal_samples
end

module WeightedMLMC = struct
  type weight_params = {
    theta: float array;
    alpha: float array;
    beta: float array;
    delta: float array;
    effort: float array;
  }

  type t = {
    levels: Estimator.level_data array;
    weights: weight_params;
    target_variance: float;
  }

  let compute_optimal_weights t =
    let n = Array.length t.levels in
    let weights = {
      theta = Array.make n 0.;
      alpha = Array.make n 0.;
      beta = Array.make n 0.;
      delta = Array.make n 0.;
      effort = Array.make n 0.;
    } in
    
    (* Base level *)
    let sigma_0 = sqrt t.levels.(0).variance in
    weights.delta.(0) <- 1.0;
    weights.alpha.(0) <- sigma_0 ** 2. /. t.target_variance ** 2.;
    weights.beta.(0) <- 0.;
    weights.effort.(0) <- sigma_0 *. sqrt t.levels.(0).cost /. t.target_variance;

    (* Compute weights for subsequent levels *)
    for l = 1 to n - 1 do
      let sigma_l = sqrt t.levels.(l).variance in
      let sigma_l_1 = sqrt t.levels.(l-1).variance in
      let rho_l = t.levels.(l).correlation in
      let mu_l = sqrt (t.levels.(l-1).cost /. t.levels.(l).cost) in
      
      if abs_float rho_l > weights.delta.(l-1) *. mu_l then begin
        let delta_l = sigma_l *. sqrt (1. -. rho_l ** 2.) *.
          sqrt (1. -. (weights.delta.(l-1) *. mu_l) ** 2.) in
          
        let theta_l = rho_l *. sigma_l /. sigma_l_1 -. 
          (if rho_l > 0. then 1. else -1.) *. delta_l *. t.target_variance *. 
          weights.effort.(l-1) /. (sigma_l_1 ** 2. *. sqrt t.levels.(l).cost) in
          
        let e_l = (delta_l *. sqrt t.levels.(l).cost +. 
          abs_float theta_l *. weights.effort.(l-1) *. t.target_variance) /. 
          t.target_variance in
          
        weights.theta.(l) <- theta_l;
        weights.delta.(l) <- e_l /. (sigma_l *. sqrt t.levels.(l).cost /. t.target_variance);
        weights.alpha.(l) <- e_l *. delta_l /. (t.target_variance *. sqrt t.levels.(l).cost);
        weights.beta.(l) <- e_l *. abs_float theta_l /. weights.effort.(l-1);
        weights.effort.(l) <- e_l
      end else begin
        weights.theta.(l) <- 0.;
        weights.delta.(l) <- 1.;
        weights.alpha.(l) <- sigma_l ** 2. /. t.target_variance ** 2.;
        weights.beta.(l) <- 0.;
        weights.effort.(l) <- sigma_l *. sqrt t.levels.(l).cost /. t.target_variance
      end
    done;
    weights

  let create sde scheme payoff target_var =
    let base = MLMC.create sde scheme payoff target_var in
    let weights = compute_optimal_weights 
      {levels = base.levels; weights = {
        theta = [||]; alpha = [||]; beta = [||]; 
        delta = [||]; effort = [||]
      }; target_variance = target_var} in
    {levels = base.levels; weights; target_variance = target_var}

  let estimate t =
    let n = Array.length t.levels in
    let estimates = Array.make n 0. in
    
    (* Base level *)
    estimates.(0) <- Estimator.estimate t.levels.(0).estimator 
      (int_of_float (ceil t.weights.alpha.(0)));
    
    (* Higher levels with weights *)
    for l = 1 to n - 1 do
      let samples = int_of_float (ceil t.weights.alpha.(l)) in
      let curr_est = Estimator.estimate t.levels.(l).estimator samples in
      estimates.(l) <- t.weights.theta.(l) *. curr_est +. 
        t.weights.beta.(l) *. estimates.(l-1)
    done;
    
    estimates.(n-1)
end

module MultiIndex = struct
  type t = int array

  let zero dim = Array.make dim 0

  let ( <= ) a b =
    try
      Array.for_all2 (fun x y -> x <= y) a b
    with Invalid_argument _ -> false
    
  let ( < ) a b =
    (a <= b) && (a <> b)
    
  let box_minus idx =
    let dim = Array.length idx in
    let result = ref [] in
    for i = 0 to dim - 1 do
      if idx.(i) > 0 then begin
        let new_idx = Array.copy idx in
        new_idx.(i) <- new_idx.(i) - 1;
        result := new_idx :: !result
      end
    done;
    !result
    
  let box_plus idx =
    let dim = Array.length idx in
    let result = ref [] in
    for i = 0 to dim - 1 do
      let new_idx = Array.copy idx in
      new_idx.(i) <- new_idx.(i) + 1;
      result := new_idx :: !result
    done;
    !result

  let min_entry idx =
    Array.fold_left min max_int idx
end

module MultiIndexMLMC = struct
  type t = {
    dim: int;
    estimators: Estimator.level_data array;
    target_variance: float;
  }

  (* Helper functions for multi-index sampling *)
  let compute_epsilon lambda lambda_prime =
    let diff = Array.map2 (-) lambda lambda_prime in
    let one_norm = Array.fold_left (+) 0 diff in
    float_of_int (if one_norm mod 2 = 0 then -1 else 1)

  (* Generate index set following Section 3 *)
  let generate_index_set dim max_level =
    let rec generate_indices curr_idx acc =
      let total_level = Array.fold_left (+) 0 curr_idx in
      if total_level <= max_level then begin
        acc := curr_idx :: !acc;
        for i = 0 to dim - 1 do
          let next_idx = Array.copy curr_idx in
          next_idx.(i) <- next_idx.(i) + 1;
          generate_indices next_idx acc
        done
      end in
    let acc = ref [] in
    generate_indices (Array.make dim 0) acc;
    !acc

  (* Generate samples for a given index *)
  let generate_samples estimator index base_samples =
    let steps = Array.map (fun i -> base_samples lsl i) index in
    let dt = Array.map (fun s -> 1. /. float_of_int s) steps in
    
    (* Generate paths with tensor product structure *)
    let generate_path () =
      let dim = Array.length index in
      let paths = Array.init dim (fun d ->
        let noise = Torch.Tensor.randn [steps.(d)] in
        Discretization.simulate_path 
          estimator.Estimator.sde
          estimator.Estimator.scheme
          steps.(d)
          dt.(d)
          noise
      ) in
      paths
    in
    
    Array.init base_samples (fun _ -> generate_path ())

  let compute_difference index samples =
    let box_minus = MultiIndex.box_minus index in
    List.fold_left (fun acc lower_idx ->
      let epsilon = compute_epsilon index lower_idx in
      acc +. epsilon *. Estimator.single_path_value samples
    ) (Estimator.single_path_value samples) box_minus

  let create dim sde scheme payoff target_var =
    let base_samples = 100 in  (* Initial sample size *)
    let max_level = 4 in       (* Maximum level per dimension *)
    
    (* Generate multi-index set *)
    let indices = generate_index_set dim max_level in
    
    (* Create estimators for each index *)
    let estimators = List.map (fun idx ->
      let est = Estimator.create sde payoff scheme 
        (Array.fold_left max 0 idx) base_samples in
      let samples = generate_samples est idx base_samples in
      let variance = Stats.variance 
        (Array.map (fun s -> compute_difference idx s) samples) in
      {
        Estimator.estimator = est;
        variance;
        correlation = 1.0;  
        cost = float_of_int (Array.fold_left ( * ) 1 
          (Array.map (fun i -> base_samples lsl i) idx));
        samples = base_samples;
      }
    ) indices |> Array.of_list in
    
    { dim; estimators; target_variance = target_var }

  (* Compute optimal sample allocation *)
  let compute_optimal_samples t =
    let n = Array.length t.estimators in
    let total_work = ref 0. in
    Array.iter (fun e ->
      total_work := !total_work +. sqrt (e.variance *. e.cost)
    ) t.estimators;
    
    Array.map (fun e ->
      let n_opt = ceil (!total_work *. 
        sqrt (e.variance /. e.cost) /. t.target_variance) in
      int_of_float n_opt
    ) t.estimators

  (* Estimate expectation *)
  let estimate t =
    let optimal_samples = compute_optimal_samples t in
    let n = Array.length t.estimators in
    let results = Array.make n 0. in
    
    (* Compute estimates for each index *)
    for i = 0 to n - 1 do
      let est = t.estimators.(i) in
      let samples = generate_samples 
        est.estimator (MultiIndex.zero t.dim) optimal_samples.(i) in
      results.(i) <- Array.fold_left (fun acc sample ->
        acc +. compute_difference (MultiIndex.zero t.dim) sample
      ) 0. samples /. float_of_int optimal_samples.(i)
    done;
    
    (* Combine results *)
    Array.fold_left (+.) 0. results
end

module WeightedMultiIndexMLMC = struct
  type multi_weight = {
    theta: float array array;  (* θᵢⱼ values *)
    alpha: float;
    beta: float;
    delta: float;
  }

  type t = {
    dim: int;
    estimators: Estimator.level_data array;
    weights: multi_weight array;
    target_variance: float;
  }

  let compute_r_matrix t indices =
    let n = List.length indices in
    let r = Array.make_matrix n n 0. in
    
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let idx_i = List.nth indices i in
        let idx_j = List.nth indices j in
        let common_indices = ref [] in
        
        (* Find common lower indices *)
        List.iter (fun idx ->
          if MultiIndex.(idx <= idx_i) && MultiIndex.(idx <= idx_j) then
            common_indices := idx :: !common_indices
        ) indices;
        
        (* Compute R value *)
        r.(i).(j) <- List.fold_left (fun acc idx ->
          let k = List.find_index ((=) idx) indices in
          let variance = t.estimators.(k).variance in
          let beta_k = t.weights.(k).beta in
          let alpha_k = t.weights.(k).alpha in
          acc +. variance *. beta_k /. alpha_k
        ) 0. !common_indices
      done
    done;
    r

  let minimize_work_variance t lambda r =
    let n = Array.length r in
    let objective theta =
      (* First term: η_λΔ_λ *)
      let delta_l = sqrt (Array.fold_left (fun acc t ->
        acc +. t *. t) 0. theta) in
      let cost_term = float_of_int (Array.fold_left ( * ) 1 
        (Array.map (fun i -> 2 lsl i) lambda)) in
      
      (* Second term: η̂_λΔ̂_λ *)
      let r_term = sqrt (Array.fold_left2 (fun acc row t1 ->
        acc +. Array.fold_left2 (fun acc' r t2 ->
          acc' +. r *. t1 *. t2
        ) 0. row theta
      ) 0. r theta) in
      
      cost_term *. delta_l +. sqrt (Array.fold_left (fun acc e ->
        acc +. e.Estimator.cost) 0. t.estimators) *. r_term
    in
    
    (* Use gradient descent to minimize objective *)
    let grad_descent init_theta =
      let max_iter = 1000 in
      let learning_rate = 0.01 in
      let theta = Array.copy init_theta in
      
      for _ = 1 to max_iter do
        let grad = Array.make n 0. in
        (* Compute gradient *)
        for i = 0 to n - 1 do
          let h = sqrt epsilon_float in
          let theta_plus = Array.copy theta in
          theta_plus.(i) <- theta_plus.(i) +. h;
          let theta_minus = Array.copy theta in
          theta_minus.(i) <- theta_minus.(i) -. h;
          grad.(i) <- (objective theta_plus -. 
                      objective theta_minus) /. (2. *. h)
        done;
        
        (* Update theta *)
        Array.iteri (fun i g ->
          theta.(i) <- theta.(i) -. learning_rate *. g
        ) grad
      done;
      theta
    in
    
    grad_descent (Array.make n 1.0)
end

module Optimization = struct
  type objective = {
    f: float array -> float;
    grad: float array -> float array;
  }

  (* LBFGS implementation *)
  module LBFGS = struct
    type memory = {
      s: float array array;  (* Parameter differences *)
      y: float array array;  (* Gradient differences *)
      rho: float array;      (* 1 / (y_i^T s_i) *)
      k: int;               (* Current iteration *)
      m: int;               (* Memory size *)
    }

    let create_memory size m = {
      s = Array.make m (Array.make size 0.);
      y = Array.make m (Array.make size 0.);
      rho = Array.make m 0.;
      k = 0;
      m = m;
    }

    let update_memory mem x_new g_new x_old g_old =
      let idx = mem.k mod mem.m in
      let s = Array.map2 (-.) x_new x_old in
      let y = Array.map2 (-.) g_new g_old in
      let rho = 1. /. Array.fold_left2 ( +. ) 0. y s in
      mem.s.(idx) <- s;
      mem.y.(idx) <- y;
      mem.rho.(idx) <- rho;
      {mem with k = mem.k + 1}

    let two_loop_recursion mem g =
      let n = Array.length g in
      let q = Array.copy g in
      let alpha = Array.make mem.m 0. in
      
      (* First loop *)
      for i = mem.k - 1 downto max 0 (mem.k - mem.m) do
        let j = i mod mem.m in
        alpha.(j) <- mem.rho.(j) *. 
          Array.fold_left2 ( +. ) 0. mem.s.(j) q;
        for k = 0 to n - 1 do
          q.(k) <- q.(k) -. alpha.(j) *. mem.y.(j).(k)
        done
      done;

      (* Scale with initial Hessian approximation *)
      let h0 = if mem.k > 0 then
        let j = (mem.k - 1) mod mem.m in
        Array.fold_left2 ( +. ) 0. mem.y.(j) mem.s.(j) /.
        Array.fold_left2 ( +. ) 0. mem.y.(j) mem.y.(j)
      else 1. in
      
      let r = Array.map (( *. ) h0) q in
      
      (* Second loop *)
      for i = max 0 (mem.k - mem.m) to mem.k - 1 do
        let j = i mod mem.m in
        let beta = mem.rho.(j) *. 
          Array.fold_left2 ( +. ) 0. mem.y.(j) r in
        for k = 0 to n - 1 do
          r.(k) <- r.(k) +. mem.s.(j).(k) *. (alpha.(j) -. beta)
        done
      done;
      
      Array.map (fun x -> -.x) r

    (* Line search satisfying Wolfe conditions *)
    let line_search obj x p =
      let c1 = 1e-4 and c2 = 0.9 in
      let rec search alpha alpha_prev f_prev g_prev =
        let x_new = Array.map2 (fun xi pi -> xi +. alpha *. pi) x p in
        let f_new = obj.f x_new in
        let g_new = obj.grad x_new in
        let dg_new = Array.fold_left2 ( +. ) 0. g_new p in
        
        if f_new > f_prev +. c1 *. alpha *. g_prev then
          search (alpha *. 0.5) alpha f_new dg_new
        else if dg_new < c2 *. g_prev then
          search (alpha *. 2.) alpha f_new dg_new
        else
          (alpha, x_new, g_new)
      in
      let f0 = obj.f x in
      let g0 = obj.grad x in
      let dg0 = Array.fold_left2 ( +. ) 0. g0 p in
      search 1. 0. f0 dg0

    let minimize ?(max_iter=1000) ?(tol=1e-6) obj x0 =
      let n = Array.length x0 in
      let mem = create_memory n 10 in
      let x = Array.copy x0 in
      let g = obj.grad x in
      
      let rec iterate mem x g iter =
        if iter >= max_iter then x
        else
          let p = two_loop_recursion mem g in
          let (alpha, x_new, g_new) = line_search obj x p in
          if Array.fold_left (fun acc xi -> 
              acc +. xi *. xi) 0. g_new < tol then x_new
          else
            let mem' = update_memory mem x_new g_new x g in
            iterate mem' x_new g_new (iter + 1)
      in
      iterate mem x g 0
  end

  let trust_region_optimize obj x0 =
    let n = Array.length x0 in
    let max_iter = 1000 in
    let initial_radius = 1.0 in
    let eta = 0.1 in
    let x = Array.copy x0 in
    let radius = ref initial_radius in
    
    for iter = 1 to max_iter do
      let g = obj.grad x in
      let b = Array.make n 0. in
      let h = Array.make_matrix n n 0. in
      
      (* Build quadratic model *)
      for i = 0 to n - 1 do
        let ei = Array.make n 0. in
        ei.(i) <- sqrt epsilon_float;
        let gi_plus = obj.grad (Array.map2 (+.) x ei) in
        let gi_minus = obj.grad (Array.map2 (-.) x ei) in
        for j = 0 to n - 1 do
          h.(i).(j) <- (gi_plus.(j) -. gi_minus.(j)) /. 
            (2. *. sqrt epsilon_float)
        done
      done;
      
      let step = Array.make n 0. in
      
      let model_reduction = 
        -. (Array.fold_left2 ( +. ) 0. g step) -. 
        0.5 *. Array.fold_left2 (fun acc si row ->
          acc +. si *. Array.fold_left2 ( *. ) 0. row step
        ) 0. step h in
      
      let x_new = Array.map2 (+.) x step in
      let actual_reduction = obj.f x -. obj.f x_new in
      let rho = actual_reduction /. model_reduction in
      
      if rho < 0.25 then
        radius := !radius /. 4.
      else if rho > 0.75 && 
              Array.fold_left (fun acc si -> max acc (abs_float si)) 
                0. step = !radius then
        radius := !radius *. 2.;
      
      (* Accept or reject step *)
      if rho > eta then
        Array.blit x_new 0 x 0 n
    done;
    x

  (* Coordinate descent implementation *)
  let coordinate_descent obj x0 =
    let n = Array.length x0 in
    let max_iter = 1000 in
    let x = Array.copy x0 in
    let tol = 1e-6 in
    
    for iter = 1 to max_iter do
      let changed = ref false in
      
      (* Cycle through coordinates *)
      for i = 0 to n - 1 do
        let xi_old = x.(i) in
        let g = (obj.grad x).(i) in
        
        (* Line search in coordinate direction *)
        let step = ref (if abs_float g > 1e-10 then -. g else 0.1) in
        while abs_float !step > tol do
          x.(i) <- xi_old +. !step;
          let f_new = obj.f x in
          let f_old = obj.f (Array.init n (fun j -> 
            if j = i then xi_old else x.(j))) in
          
          if f_new < f_old then begin
            changed := true;
            step := !step *. 1.2
          end else begin
            x.(i) <- xi_old;
            step := !step *. 0.5
          end
        done
      done;
      
      if not !changed then
        iter := max_iter
    done;
    x
end

module PathGeneration = struct
  type path_features = {
    mean_reversion: float option;
    jumps: (float * float) list;
    seasonality: float -> float;
    volatility_clustering: bool;
  }

  (* Helper for Brownian bridge construction *)
  module BrownianBridge = struct
    type bridge_point = {
      time: float;
      value: float;
      left_idx: int option;
      right_idx: int option;
    }

    let construct_bridge points =
      let n = Array.length points in
      let bridge = Array.make n 
        {time=0.; value=0.; left_idx=None; right_idx=None} in
      
      (* Initialize endpoints *)
      bridge.(0) <- points.(0);
      bridge.(n-1) <- points.(n-1);
      
      (* Build bridge recursively *)
      let rec fill_points left right =
        if right - left > 1 then begin
          let mid = (left + right) / 2 in
          let t_left = bridge.(left).time in
          let t_right = bridge.(right).time in
          let t_mid = (t_left +. t_right) /. 2. in
          
          let w_left = bridge.(left).value in
          let w_right = bridge.(right).value in
          
          (* Compute conditional expectation and variance *)
          let lambda = (t_right -. t_mid) /. (t_right -. t_left) in
          let mean = lambda *. w_left +. (1. -. lambda) *. w_right in
          let var = (t_right -. t_mid) *. (t_mid -. t_left) /. 
            (t_right -. t_left) in
          
          bridge.(mid) <- {
            time = t_mid;
            value = mean +. sqrt var *. Random.float 1.0;
            left_idx = Some left;
            right_idx = Some right;
          };
          
          fill_points left mid;
          fill_points mid right
        end
      in
      fill_points 0 (n-1);
      bridge
  end

  let generate_path sde features steps dt noise =
    let open Torch.Tensor in
    let path = zeros [steps + 1] in
    index_put_ path [Some 0] (float_vec [sde.SDE.init_price]);
    
    (* Generate bridge points *)
    let bridge_points = Array.init (steps + 1) (fun i ->
      {BrownianBridge.
        time = float_of_int i *. dt;
        value = get noise i |> float_value;
        left_idx = None;
        right_idx = None;
      }
    ) in
    
    let bridge = BrownianBridge.construct_bridge bridge_points in
    
    (* Generate path using bridge *)
    for i = 1 to steps do
      let curr_price = get path (i-1) |> float_value in
      
      (* Base drift and volatility *)
      let base_drift = SDE.drift sde.SDE.model curr_price in
      let base_vol = SDE.volatility sde.SDE.model curr_price in
      
      (* Add mean reversion *)
      let mr_drift = match features.mean_reversion with
      | Some speed -> 
          base_drift +. speed *. (sde.SDE.init_price -. curr_price)
      | None -> base_drift in
      
      (* Add seasonality *)
      let t = float_of_int i *. dt in
      let seasonal = features.seasonality t in
      
      (* Add volatility clustering *)
      let vol_factor = 
        if features.volatility_clustering && i > 1 then
          let prev_return = abs_float (curr_price -. 
            (get path (i-2) |> float_value)) /. curr_price in
          if prev_return > 2. *. base_vol *. sqrt dt 
          then 1.5 
          else 1.0
        else 1.0 in
      
      (* Compute next price *)
      let bridge_incr = bridge.(i).value -. bridge.(i-1).value in
      let next_price = curr_price +. 
        mr_drift *. seasonal *. dt +.
        base_vol *. vol_factor *. bridge_incr in
      
      (* Add jumps *)
      let with_jumps = List.fold_left (fun price (jump_time, jump_size) ->
        if abs_float (t -. jump_time) < dt /. 2. then
          price *. (1. +. jump_size)
        else price
      ) next_price features.jumps in
      
      index_put_ path [Some i] (float_vec [with_jumps])
    done;
    path
end

module NumericalSchemes = struct
  module MultiStep = struct
    type step_config = {
      order: int;
      stability_factor: float;
      error_tolerance: float;
      max_steps: int;
    }

    (* Adams-Bashforth coefficients *)
    let get_ab_coefficients order =
      match order with
      | 1 -> [|1.|]
      | 2 -> [|1.5; -0.5|]
      | 3 -> [|23./.12.; -16./.12.; 5./.12.|]
      | 4 -> [|55./.24.; -59./.24.; 37./.24.; -9./.24.|]
      | _ -> failwith "Unsupported order"

    (* Adams-Moulton coefficients *)
    let get_am_coefficients order =
      match order with
      | 1 -> [|1.|]
      | 2 -> [|1./.2.; 1./.2.|]
      | 3 -> [|5./.12.; 8./.12.; -1./.12.|]
      | 4 -> [|9./.24.; 19./.24.; -5./.24.; 1./.24.|]
      | _ -> failwith "Unsupported order"

    (* BDF coefficients *)
    let get_bdf_coefficients order =
      match order with
      | 1 -> [|1.; -1.|]
      | 2 -> [|3./.2.; -2.; 1./.2.|]
      | 3 -> [|11./.6.; -3.; 3./.2.; -1./.3.|]
      | 4 -> [|25./.12.; -4.; 3.; -4./.3.; 1./.4.|]
      | _ -> failwith "Unsupported order"

    let adams_bashforth config sde state history =
      let open Torch.Tensor in
      let h = sde.dt in
      let coeffs = get_ab_coefficients config.order in
      
      let fn x = SDE.drift sde.SDE.model x in
      let sum = ref 0. in
      Array.iteri (fun i c ->
        sum := !sum +. c *. fn (get history.(i) 0 |> float_value)
      ) coeffs;
      
      float_vec [float_value state +. h *. !sum]

    let adams_moulton config sde state history =
      let open Torch.Tensor in
      let h = sde.dt in
      let coeffs = get_am_coefficients config.order in
      
      let fn x = SDE.drift sde.SDE.model x in
      
      (* Implicit solution using fixed-point iteration *)
      let rec iterate guess iter =
        if iter >= config.max_steps then guess
        else
          let sum = ref (coeffs.(0) *. fn (float_value guess)) in
          Array.iteri (fun i c ->
            if i > 0 then
              sum := !sum +. c *. fn (get history.(i-1) 0 |> float_value)
          ) coeffs;
          
          let next = float_value state +. h *. !sum in
          if abs_float (next -. float_value guess) < config.error_tolerance
          then float_vec [next]
          else iterate (float_vec [next]) (iter + 1)
      in
      
      iterate state 0

    let bdf config sde state history =
      let open Torch.Tensor in
      let h = sde.dt in
      let coeffs = get_bdf_coefficients config.order in
      
      let fn x = SDE.drift sde.SDE.model x in
      
      (* Newton iteration for implicit solution *)
      let rec newton_solve guess iter =
        if iter >= config.max_steps then guess
        else
          let f = coeffs.(0) *. float_value guess in
          Array.iteri (fun i c ->
            if i > 0 then
              f <- f +. c *. get history.(i-1) 0 |> float_value
          ) coeffs;
          f <- f -. h *. fn (float_value guess);
          
          let df = coeffs.(0) -. h *. 
            SDE.drift_derivative sde.SDE.model (float_value guess) in
          
          let next = float_value guess -. f /. df in
          if abs_float (next -. float_value guess) < config.error_tolerance
          then float_vec [next]
          else newton_solve (float_vec [next]) (iter + 1)
      in
      
      newton_solve state 0
  end

  module PathIntegral = struct
    type integral_method = 
      | Trapezoidal 
      | Simpson 
      | GaussKronrod 
      | AdaptiveQuad

    let integrate method_ path timesteps =
      match method_ with
      | Trapezoidal ->
          let open Torch.Tensor in
          let n = size path 0 in
          let sum = ref 0. in
          for i = 0 to n - 2 do
            let y1 = get path i |> float_value in
            let y2 = get path (i + 1) |> float_value in
            sum := !sum +. (y1 +. y2) *. timesteps.(i) /. 2.
          done;
          !sum

      | Simpson ->
          let open Torch.Tensor in
          let n = size path 0 in
          let sum = ref 0. in
          for i = 0 to (n-3)/2 do
            let y1 = get path (2*i) |> float_value in
            let y2 = get path (2*i + 1) |> float_value in
            let y3 = get path (2*i + 2) |> float_value in
            let h = timesteps.(2*i) +. timesteps.(2*i + 1) in
            sum := !sum +. h /. 6. *. (y1 +. 4. *. y2 +. y3)
          done;
          !sum

      | GaussKronrod ->
        let nodes = [|-0.991455371120813; -0.949107912342759;
                     -0.864864423359769; -0.741531185599394;
                     -0.586087235467691; -0.405845151377397;
                     -0.207784955007898; 0.0;
                     0.207784955007898; 0.405845151377397;
                     0.586087235467691; 0.741531185599394;
                     0.864864423359769; 0.949107912342759;
                     0.991455371120813|] in
        let weights = [|0.022935322010529; 0.063092092629979;
                       0.104790010322250; 0.140653259715525;
                       0.169004726639267; 0.190350578064785;
                       0.204432940075298; 0.209482141084728;
                       0.204432940075298; 0.190350578064785;
                       0.169004726639267; 0.140653259715525;
                       0.104790010322250; 0.063092092629979;
                       0.022935322010529|] in
        
        let open Torch.Tensor in
        let n = size path 0 in
        let sum = ref 0. in
        
        for i = 0 to n - 2 do
          let a = get path i |> float_value in
          let b = get path (i + 1) |> float_value in
          let h = timesteps.(i) in
          
          let segment_sum = ref 0. in
          Array.iteri (fun j node ->
            let t = (a +. b) /. 2. +. (b -. a) /. 2. *. node in
            segment_sum := !segment_sum +. weights.(j) *. t
          ) nodes;
          
          sum := !sum +. (b -. a) /. 2. *. !segment_sum
        done;
        !sum

    | AdaptiveQuad ->
        let open Torch.Tensor in
        let tolerance = 1e-6 in
        
        (* Recursive adaptive quadrature *)
        let rec adaptive a b tol =
          let mid = (a +. b) /. 2. in
          let fa = get path (int_of_float (a *. float_of_int (size path 0))) 
                  |> float_value in
          let fm = get path (int_of_float (mid *. float_of_int (size path 0))) 
                  |> float_value in
          let fb = get path (int_of_float (b *. float_of_int (size path 0))) 
                  |> float_value in
          
          (* Simpson's rule on whole interval and subintervals *)
          let whole = (b -. a) /. 6. *. (fa +. 4. *. fm +. fb) in
          let left = (mid -. a) /. 6. *. 
            (fa +. 4. *. ((fa +. fm) /. 2.) +. fm) in
          let right = (b -. mid) /. 6. *. 
            (fm +. 4. *. ((fm +. fb) /. 2.) +. fb) in
          
          if abs_float (whole -. (left +. right)) < tol then
            whole
          else
            adaptive a mid (tol /. 2.) +. adaptive mid b (tol /. 2.)
        in
        
        let n = size path 0 in
        adaptive 0. (float_of_int (n - 1)) tolerance
  end

  module EnhancedPathGeneration = struct
    type path_features = {
      mean_reversion: float option;
      jumps: (float * float) list;
      seasonality: float -> float;
      volatility_clustering: bool;
    }

    type generation_config = {
      base_scheme: Discretization.scheme;
      use_adaptive_timesteps: bool;
      error_tolerance: float;
      min_timestep: float;
    }

    (* Generate enhanced path with all features *)
    let generate_path sde features config steps dt noise =
      let open Torch.Tensor in
      let path = zeros [steps + 1] in
      index_put_ path [Some 0] (float_vec [sde.SDE.init_price]);
      
      let get_timestep i curr_price next_price =
        if config.use_adaptive_timesteps then
          let error_est = abs_float (next_price -. curr_price) /. curr_price in
          max config.min_timestep (dt *. min 1. (config.error_tolerance /. error_est))
        else dt in
      
      let compute_next_price i curr_price =
        (* Base drift and volatility *)
        let base_drift = SDE.drift sde.SDE.model curr_price in
        let base_vol = SDE.volatility sde.SDE.model curr_price in
        
        (* Add mean reversion if specified *)
        let mr_drift = match features.mean_reversion with
        | Some speed -> base_drift +. speed *. (sde.SDE.init_price -. curr_price)
        | None -> base_drift in
        
        (* Add seasonality *)
        let t = float_of_int i *. dt in
        let seasonal = features.seasonality t in
        
        (* Add volatility clustering *)
        let vol_factor = 
          if features.volatility_clustering && i > 0 then
            let prev_return = abs_float (curr_price -. 
              (get path (i-1) |> float_value)) /. curr_price in
            if prev_return > 2. *. base_vol *. sqrt dt then 1.5 else 1.0
          else 1.0 in
        
        (* Basic step *)
        let w = get noise i |> float_value in
        let next_price = match config.base_scheme with
        | Euler ->
            curr_price +. mr_drift *. seasonal *. dt +.
            base_vol *. vol_factor *. w
        | Milstein ->
            let vol_deriv = SDE.volatility_derivative sde.SDE.model curr_price in
            curr_price +. mr_drift *. seasonal *. dt +.
            base_vol *. vol_factor *. w +.
            0.5 *. base_vol *. vol_deriv *. vol_factor *. (w *. w -. dt) in
        
        (* Add jumps *)
        List.fold_left (fun price (jump_time, jump_size) ->
          if abs_float (t -. jump_time) < dt /. 2. then
            price *. (1. +. jump_size)
          else price
        ) next_price features.jumps in
      
      let rec step i =
        if i >= steps then path
        else begin
          let curr_price = get path i |> float_value in
          let next_price = compute_next_price i curr_price in
          let actual_dt = get_timestep i curr_price next_price in
          
          index_put_ path [Some (i + 1)] (float_vec [next_price]);
          step (i + 1)
        end in
      
      step 0

    (* Helper for interpolating between path points *)
    let interpolate path t =
      let open Torch.Tensor in
      let n = size path 0 in
      let idx = floor (t *. float_of_int (n - 1)) |> int_of_float in
      if idx >= n - 1 then
        get path (n - 1) |> float_value
      else
        let t0 = float_of_int idx /. float_of_int (n - 1) in
        let t1 = float_of_int (idx + 1) /. float_of_int (n - 1) in
        let y0 = get path idx |> float_value in
        let y1 = get path (idx + 1) |> float_value in
        y0 +. (y1 -. y0) *. (t -. t0) /. (t1 -. t0)
  end
end