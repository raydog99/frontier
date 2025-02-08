open Torch

type dim3 = {
  x: float;
  y: float;
  z: float;
}

type mctou_params = {
  kappa: float;          (* Mean reversion speed for X1 *)
  epsilon: float;        (* Fast mean reversion parameter *)
  delta: float;          (* Slow mean reversion parameter *)
  alpha2: float;         (* Mean level for X2 *)
  alpha3: float;         (* Mean level for X3 *)
  sigma1: float;         (* Volatility for X1 *)
  sigma2: float;         (* Volatility for X2 *)
  sigma3: float;         (* Volatility for X3 *)
  rho12: float;         (* Correlation between X1 and X2 *)
  rho13: float;         (* Correlation between X1 and X3 *)
  rho23: float;         (* Correlation between X2 and X3 *)
  lambda1: float;       (* Market price of risk for X1 *)
  lambda2: float;       (* Market price of risk for X2 *)
  lambda3: float;       (* Market price of risk for X3 *)
}

type state = {
  x1: Tensor.t;
  x2: Tensor.t;
  x3: Tensor.t;
}

type brownian_motion = {
  dw1: Tensor.t;
  dw2: Tensor.t;
  dw3: Tensor.t;
}

(* Numerical utilities *)
module Numerical = struct
  (* Gauss-Kronrod quadrature for numerical integration *)
  let gauss_kronrod_15 = 
    let points = [|
      -0.991455371120813; -0.949107912342759; -0.864864423359769;
      -0.741531185599394; -0.586087235467691; -0.405845151377397;
      -0.207784955007898; 0.0; 0.207784955007898; 0.405845151377397;
      0.586087235467691; 0.741531185599394; 0.864864423359769;
      0.949107912342759; 0.991455371120813
    |] in
    let weights = [|
      0.022935322010529; 0.063092092629979; 0.104790010322250;
      0.140653259715525; 0.169004726639267; 0.190350578064785;
      0.204432940075298; 0.209482141084728; 0.204432940075298;
      0.190350578064785; 0.169004726639267; 0.140653259715525;
      0.104790010322250; 0.063092092629979; 0.022935322010529
    |] in
    (points, weights)

  (* Trapezoidal integration *)
  let integrate f a b n =
    let h = (b -. a) /. float_of_int n in
    let rec sum i acc =
      if i > n then acc
      else
        let x = a +. float_of_int i *. h in
        let y = if i = 0 || i = n then f x else 2. *. f x in
        sum (i + 1) (acc +. y)
    in
    (h /. 2.) *. sum 0 0.

  (* Adaptive integration *)
  let adaptive_quadrature f a b tol =
    let rec adaptive_step f a b tol acc =
      let mid = (a +. b) /. 2. in
      let h = (b -. a) /. 2. in
      
      (* Compute Gauss and Gauss-Kronrod approximations *)
      let (xgk, wgk) = gauss_kronrod_15 in
      let gauss = ref 0. in
      let kronrod = ref 0. in
      
      Array.iteri (fun i x ->
        let t = mid +. h *. x in
        let ft = f t in
        if i mod 2 = 0 then gauss := !gauss +. wgk.(i/2) *. ft;
        kronrod := !kronrod +. wgk.(i) *. ft
      ) xgk;
      
      gauss := h *. !gauss;
      kronrod := h *. !kronrod;
      
      let error = abs_float (!kronrod -. !gauss) in
      if error < tol *. abs_float !kronrod then
        acc +. !kronrod
      else if h < 1e-14 then
        acc +. !kronrod
      else
        let left = adaptive_step f a mid (tol /. sqrt 2.) acc in
        adaptive_step f mid b (tol /. sqrt 2.) left
    in
    
    adaptive_step f a b tol 0.

  (* Matrix operations *)
  let cholesky_decomposition mat =
    let n = size mat |> Array.get in
    let l = zeros [|n; n|] in
    
    for i = 0 to n - 1 do
      for j = 0 to i do
        let s = ref (get mat [|i; j|]) in
        for k = 0 to j - 1 do
          s := !s -. get l [|i; k|] *. get l [|j; k|]
        done;
        set l [|i; j|] (
          if i = j then sqrt !s
          else !s /. get l [|j; j|]
        )
      done
    done;
    l
end

let create_correlation_matrix params 
  of_float2 [|
    [|1.0; params.rho12; params.rho13|];
    [|params.rho12; 1.0; params.rho23|];
    [|params.rho13; params.rho23; 1.0|]
  |]

let create_volatility_matrix params 
  of_float2 [|
    [|params.sigma1; 0.0; 0.0|];
    [|0.0; params.sigma2 /. sqrt params.epsilon; 0.0|];
    [|0.0; 0.0; sqrt params.delta *. params.sigma3|]
  |]

let create_drift_matrix params 
  of_float2 [|
    [|-.params.kappa; params.kappa; params.kappa|];
    [|0.0; -.1. /. params.epsilon; 0.0|];
    [|0.0; 0.0; -.params.delta|]
  |]

let create_lambda_vector params 
  of_float1 [|params.lambda1; params.lambda2; params.lambda3|]

let create_mean_vector params 
  of_float1 [|0.0; params.alpha2 /. params.epsilon; 
              params.delta *. params.alpha3|]

(* Scale-separated evolution step with Euler-Maruyama scheme *)
let euler_maruyama state params dt dw measure 
  let drift = matmul (create_drift_matrix params) 
    (cat [state.x1; state.x2; state.x3] ~dim:0 |> 
     reshape ~shape:[|-1; 1|]) in
  let mean = create_mean_vector params |> reshape ~shape:[|-1; 1|] in
  let lambda = if measure = `Q then 
    create_lambda_vector params |> reshape ~shape:[|-1; 1|]
    else zeros [|3; 1|] in
  let vol = create_volatility_matrix params in
  let corr = create_correlation_matrix params in
  
  let diffusion = matmul (matmul vol corr) 
    (cat [dw.dw1; dw.dw2; dw.dw3] ~dim:0 |> reshape ~shape:[|-1; 1|]) in
  
  let dx = add (add (add (mul drift dt) (mul mean dt)) 
                   (mul (neg lambda) dt)) diffusion in
  let new_state = split ~dim:0 ~sizes:[|1; 1; 1|] dx in
  {
    x1 = add state.x1 (index new_state 0);
    x2 = add state.x2 (index new_state 1);
    x3 = add state.x3 (index new_state 2);
  }

let evolve_p state params dt dw = euler_maruyama state params dt dw `P
let evolve_q state params dt dw = euler_maruyama state params dt dw `Q

(* Futures pricing *)
module FuturesPricing = struct
  type futures_contract = {
    maturity: float;
    notional: float;
  }

  (* PDE solver for futures pricing *)
  module PDE = struct
    type grid_params = {
      x_min: float array;
      x_max: float array;
      nx: int array;
      dt: float;
    }

    type grid = {
      points: Tensor.t array;
      dx: float array;
      dt: float;
    }

    let create_grid params =
 
      let points = Array.mapi (fun i _ ->
        linspace ~start:params.x_min.(i) 
                ~end_:params.x_max.(i) 
                ~steps:(params.nx.(i) + 1)
      ) params.x_min in
      
      let dx = Array.mapi (fun i _ ->
        (params.x_max.(i) -. params.x_min.(i)) /. 
        float_of_int params.nx.(i)
      ) params.x_min in
      
      {points; dx; dt = params.dt}

    (* Finite difference operators *)
    let create_differential_operators grid =
 
      (* First derivatives - central difference *)
      let d1 = Array.mapi (fun i points ->
        let n = size points |> Array.get in
        let h = grid.dx.(i) in
        let op = zeros [|n; n|] in
        
        for j = 1 to n-2 do
          set op [|j; j-1|] (-1. /. (2. *. h));
          set op [|j; j+1|] (1. /. (2. *. h))
        done;
        op
      ) grid.points in

      (* Second derivatives *)
      let d2 = Array.mapi (fun i points ->
        let n = size points |> Array.get in
        let h = grid.dx.(i) in
        let op = zeros [|n; n|] in
        
        for j = 1 to n-2 do
          set op [|j; j-1|] (1. /. (h *. h));
          set op [|j; j|] (-2. /. (h *. h));
          set op [|j; j+1|] (1. /. (h *. h))
        done;
        op
      ) grid.points in

      (d1, d2)

    (* Apply L^Q operator *)
    let apply_lq params state u grid =
 
      let (d1, d2) = create_differential_operators grid in
      let mu = create_mean_vector params in
      let lambda = create_lambda_vector params in
      let vol = create_vol_matrix params state in
      let sigma_mat = matmul vol (transpose vol) in
      
      (* Compute first derivatives *)
      let ux = Array.mapi (fun i _ -> matmul d1.(i) u) grid.points in
      let drift_term = Array.fold_left2 (fun acc ui mui ->
        add acc (mul ui mui)
      ) (zeros (size u)) ux (Array.init 3 (fun i -> get mu [|i|])) in
      
      (* Compute second derivatives *)
      let uxx = Array.mapi (fun i _ -> matmul d2.(i) u) grid.points in
      let diff_term = ref (zeros (size u)) in
      for i = 0 to 2 do
        for j = 0 to 2 do
          diff_term := add !diff_term 
            (mul uxx.(i) (get sigma_mat [|i; j|]))
        done
      done;
      
      add drift_term (mul !diff_term 0.5)
  end

  (* Core pricing functions *)
  module Pricing = struct
    (* Compute a(k)(t) coefficient *)
    let compute_ak params t tk =
 
      let tau = tk -. t in
      let k_mat = create_drift_matrix params in
      exp (mul (neg (scalar_tensor tau)) (transpose k_mat)) |>
      matmul (of_float1 [|1.0; 0.0; 0.0|])

    (* Compute β(k)(t) using adaptive quadrature *)
    let compute_beta params t tk =
 
      let integrand s =
        let ak = compute_ak params s tk in
        let mu = create_mean_vector params in
        let vol = create_vol_matrix params {
          x1 = zeros [|1|];
          x2 = zeros [|1|];
          x3 = zeros [|1|];
        } in
        
        let drift_term = dot mu ak in
        let sigma_ak = matmul vol ak in
        let var_term = mul (dot sigma_ak sigma_ak) 0.5 in
        add drift_term var_term
      in
      
      Numerical.adaptive_quadrature 
        (fun s -> float_value (integrand s)) t tk 1e-8

    (* Compute futures price *)
    let compute_price params contract state t =
 
      if t >= contract.maturity then
        exp state.x1 |> float_value
      else
        let ak = compute_ak params t contract.maturity in
        let beta = compute_beta params t contract.maturity in
        let state_vec = cat [state.x1; state.x2; state.x3] ~dim:0 in
        exp (add (dot ak state_vec) (scalar_tensor beta)) |> float_value

    (* Compute price dynamics *)
    let compute_price_dynamics params state t =
 
      let vol = create_vol_matrix params state in
      let mu = create_mean_vector params in
      let lambda = create_lambda_vector params in
      (mu, vol)
  end
end

(* Portfolio optimization *)
module Portfolio = struct
  type portfolio_params = {
    gamma: float;         (* Risk aversion parameter *)
    contracts: FuturesPricing.futures_contract array;
    t_horizon: float;     (* Investment horizon *)
  }

  let compute_optimal_strategy params state t =
    (* Get price dynamics *)
    let (mu_f, sigma_f) = FuturesPricing.Pricing.compute_price_dynamics 
      params state t in
    
    (* Construct strategy *)
    let n = Array.length params.contracts in
    let mu_f_vec = reshape mu_f [|n|] in
    let sigma_f_mat = reshape sigma_f [|n; 3|] in
    
    (* Compute optimal weights *)
    let sigma_sigma_t = matmul sigma_f_mat (transpose sigma_f_mat) in
    let inv_sigma = inverse sigma_sigma_t in
    div (matmul inv_sigma mu_f_vec) params.gamma

  (* Simulate wealth process *)
  let simulate_wealth params state strategy t dt dw =
    let (mu_f, sigma_f) = FuturesPricing.Pricing.compute_price_dynamics 
      params state t in
    
    (* Compute wealth change *)
    let drift = dot strategy mu_f in
    let diff = matmul sigma_f (cat [dw.dw1; dw.dw2; dw.dw3] ~dim:0) in
    let diff_term = dot strategy diff in
    
    add (mul drift (scalar_tensor dt)) diff_term |> float_value
end

(* Portfolio constraints and transaction costs *)
module PortfolioConstraints = struct
  type constraints = {
    position_limits: (float * float) array;
    total_exposure: float option;
    leverage: float option;
  }

  type transaction_costs = {
    fixed: float;
    proportional: float;
    market_impact: float;
  }

  (* Project strategy onto constraints *)
  let projection strategy constraints =
    (* Position limits *)
    let projected = copy strategy in
    let n = size strategy |> Array.get in
    
    for i = 0 to n-1 do
      let pos = float_value (slice strategy [|Index.Slice(Some i, Some(i+1), None)|]) in
      let (min_pos, max_pos) = constraints.position_limits.(i) in
      let new_pos = max min_pos (min max_pos pos) in
      set projected [|i|] new_pos
    done;
    
    (* Total exposure *)
    let project_exposure pi =
      match constraints.total_exposure with
      | Some limit ->
          let norm_pi = norm pi |> float_value in
          if norm_pi > limit then
            mul pi (scalar_tensor (limit /. norm_pi))
          else pi
      | None -> pi
    in
    
    (* Leverage constraint *)
    let project_leverage pi =
      match constraints.leverage with
      | Some limit ->
          let sum_abs = sum (abs pi) |> float_value in
          if sum_abs > limit then
            mul pi (scalar_tensor (limit /. sum_abs))
          else pi
      | None -> pi
    in
    
    projected |> project_exposure |> project_leverage

  (* Compute transaction costs *)
  let compute_transaction_costs params old_strategy new_strategy prices =
    (* Trade sizes *)
    let trades = sub new_strategy old_strategy in
    let notional = mul trades prices in
    
    (* Fixed costs *)
    let n_trades = sum (abs trades |> gt (scalar_tensor 1e-6)) |> float_value in
    let fixed = params.fixed *. n_trades in
    
    (* Proportional costs *)
    let prop = sum (abs notional) |> float_value in
    let prop_costs = prop *. params.proportional in
    
    (* Market impact *)
    let impact = sum (mul (abs notional) notional) |> float_value in
    let impact_costs = impact *. params.market_impact in
    
    fixed +. prop_costs +. impact_costs
end

(* HJB *)
module HJB = struct
  type hjb_params = {
    grid_w: float array;    (* Wealth grid points *)
    nw: int;                (* Number of wealth grid points *)
    nt: int;                (* Number of time steps *)
    theta: float;           (* Time discretization parameter *)
  }

  (* Value function computation *)
  module Value_Function = struct
    let terminal_condition params w =
      neg (exp (mul (scalar_tensor (-.params.gamma)) (scalar_tensor w)))

    (* Build wealth operators *)
    let build_wealth_operators params =
      let dw = (params.grid_w.(params.nw-1) -. params.grid_w.(0)) /. 
               float_of_int params.nw in
      
      (* First derivative *)
      let d1 = zeros [|params.nw; params.nw|] in
      for i = 1 to params.nw-2 do
        set d1 [|i; i-1|] (-1. /. (2. *. dw));
        set d1 [|i; i+1|] (1. /. (2. *. dw))
      done;
      
      (* Second derivative *)
      let d2 = zeros [|params.nw; params.nw|] in
      for i = 1 to params.nw-2 do
        set d2 [|i; i-1|] (1. /. (dw *. dw));
        set d2 [|i; i|] (-2. /. (dw *. dw));
        set d2 [|i; i+1|] (1. /. (dw *. dw))
      done;
      
      (d1, d2, dw)

    (* Compute value function *)
    let compute_value_function params state t w =
      (* Get optimal strategy *)
      let strategy = Portfolio.compute_optimal_strategy params state t in
      
      (* Get price dynamics *)
      let (mu_f, sigma_f) = FuturesPricing.Pricing.compute_price_dynamics 
        params state t in
      
      (* Compute Λ²(t) *)
      let sigma_ff = matmul sigma_f (transpose sigma_f) in
      let inv_sigma = inverse sigma_ff in
      let lambda_squared = dot mu_f (matmul inv_sigma mu_f) |> float_value in
      
      (* Compute integral *)
      let tau = params.t_horizon -. t in
      -. exp (-. params.gamma *. (w +. lambda_squared *. tau /. 
              (2. *. params.gamma)))
  end

  (* HJB equation solver *)
  module Solver = struct
    let apply_hjb_operator params state t w v d1 d2 =
      (* Get optimal strategy and price dynamics *)
      let strategy = Portfolio.compute_optimal_strategy params state t in
      let (mu_f, sigma_f) = FuturesPricing.Pricing.compute_price_dynamics 
        params state t in
      
      (* Compute drift term *)
      let v_w = matmul d1 v in
      let drift = dot strategy mu_f in
      let drift_term = mul v_w drift in
      
      (* Compute diffusion term *)
      let v_ww = matmul d2 v in
      let sigma_pi = matmul sigma_f strategy in
      let diffusion = dot sigma_pi sigma_pi in
      let diff_term = mul v_ww (mul diffusion (scalar_tensor 0.5)) in
      
      add drift_term diff_term

    (* Solve HJB equation *)
    let solve params state t t_end =
      let dt = (t_end -. t) /. float_of_int params.nt in
      let (d1, d2, dw) = Value_Function.build_wealth_operators params in
      
      (* Initialize with terminal condition *)
      let v = Array.init params.nw (fun i ->
        let w_i = params.grid_w.(0) +. float_of_int i *. dw in
        Value_Function.terminal_condition params w_i
      ) |> stack ~dim:0 in
      
      (* Time stepping *)
      let rec step v_curr t_curr =
        if t_curr <= t then v_curr
        else
          let t_next = t_curr -. dt in
          
          (* Build linear system *)
          let a = eye params.nw in
          let b = copy v_curr in
          
          (* Apply HJB operator *)
          let hjb_term = apply_hjb_operator params state t_curr 
            params.grid_w.(0) v_curr d1 d2 in
          
          (* Update solution *)
          let v_next =
            if params.theta > 0. then
              (* Implicit step *)
              let lhs = add a (mul hjb_term 
                            (scalar_tensor (params.theta *. dt))) in
              gesv lhs b |> snd
            else
              (* Explicit step *)
              add v_curr (mul hjb_term (scalar_tensor dt))
          in
          
          (* Apply boundary conditions *)
          let v_next = copy v_next in
          set v_next [|0|] (get v_next [|1|]);
          set v_next [|params.nw-1|] (get v_next [|params.nw-2|]);
          
          step v_next t_next
      in
      
      step v t_end
  end

  (* Solution verification *)
  module Verification = struct
    type verification_result = {
      viscosity_subsolution: bool;
      viscosity_supersolution: bool;
      boundary_conditions: bool;
      monotonicity: bool;
      concavity: bool;
    }

    (* Verify viscosity solution properties *)
    let verify_viscosity_solution params state t w solution =
 
      (* Test function *)
      let test_function x0 p =
        fun x -> add (dot p (sub x x0))
                   (mul (norm (sub x x0)) (scalar_tensor 2.))
      in
      
      (* Check subsolution property *)
      let verify_subsolution x0 =
        let p = randn [|3|] in
        let phi = test_function x0 p in
        
        let phi_t = (phi (add x0 (scalar_tensor 0.01)) -. 
                    phi x0) /. 0.01 in
        let phi_x = grad phi x0 in
        let phi_xx = hessian phi x0 in
        
        let h = Solver.apply_hjb_operator params state t w 
                  phi_t phi_x phi_xx in
        float_value h >= -1e-6
      in
      
      (* Check supersolution property *)
      let verify_supersolution x0 =
        let p = randn [|3|] in
        let phi = test_function x0 p in
        
        let phi_t = (phi (add x0 (scalar_tensor 0.01)) -. 
                    phi x0) /. 0.01 in
        let phi_x = grad phi x0 in
        let phi_xx = hessian phi x0 in
        
        let h = Solver.apply_hjb_operator params state t w 
                  phi_t phi_x phi_xx in
        float_value h <= 1e-6
      in
      
      (* Verify properties *)
      let test_points = Array.init 10 (fun _ -> randn [|3|]) in
      let subsolution = Array.for_all verify_subsolution test_points in
      let supersolution = Array.for_all verify_supersolution test_points in
      
      (* Check monotonicity and concavity *)
      let v = Value_Function.compute_value_function params state t w in
      let v_up = Value_Function.compute_value_function params state t (w +. 0.1) in
      let v_down = Value_Function.compute_value_function params state t (w -. 0.1) in
      
      let monotone = v_up > v && v > v_down in
      let concave = (v_up +. v_down) /. 2. <= v in
      
      (* Check boundary conditions *)
      let boundary_valid =
        let w_min = params.grid_w.(0) in
        let w_max = params.grid_w.(params.nw-1) in
        let v_min = Value_Function.compute_value_function params state t w_min in
        let v_max = Value_Function.compute_value_function params state t w_max in
        abs_float v_min < 1e10 && v_max <= 0.
      in
      
      {
        viscosity_subsolution = subsolution;
        viscosity_supersolution = supersolution;
        boundary_conditions = boundary_valid;
        monotonicity = monotone;
        concavity = concave;
      }
  end
end

(* Risk management *)
module RiskManagement = struct
  type risk_metrics = {
    var: float;              (* Value at Risk *)
    cvar: float;             (* Conditional Value at Risk *)
    volatility: float;       (* Portfolio volatility *)
    sharpe_ratio: float;     (* Sharpe ratio *)
  }

  type risk_decomposition = {
    systematic: Tensor.t;     (* Systematic risk *)
    idiosyncratic: Tensor.t;  (* Idiosyncratic risk *)
    basis: Tensor.t;          (* Basis risk *)
  }

  (* Compute risk metrics *)
  let compute_risk_metrics params state t strategy n_paths =
    (* Simulate paths *)
    let dt = 0.01 in
    let steps = int_of_float ((params.t_horizon -. t) /. dt) in
    
    let simulate_path () =
      let rec loop state_t wealth acc i =
        if i >= steps then wealth :: acc
        else
          let dw = MCTOU_Extension.simulate_brownian_increments params dt `P in
          let new_state = MCTOU.evolve_p state_t params dt dw in
          let dwealth = Portfolio.simulate_wealth params state_t strategy t dt dw in
          loop new_state (wealth +. dwealth) (wealth :: acc) (i + 1)
      in
      loop state 0. [] 0
    in
    
    (* Generate paths *)
    let paths = Array.init n_paths (fun _ -> simulate_path ()) in
    
    (* Compute metrics *)
    let terminal_wealth = Array.map List.hd paths in
    Array.sort compare terminal_wealth;
    
    let var_95 = terminal_wealth.(int_of_float (0.05 *. float_of_int n_paths)) in
    
    let cvar_95 = 
      let cutoff = int_of_float (0.05 *. float_of_int n_paths) in
      let sum = Array.fold_left (+.) 0. 
        (Array.sub terminal_wealth 0 cutoff) in
      sum /. float_of_int cutoff
    in
    
    let mean = Array.fold_left (+.) 0. terminal_wealth /. 
               float_of_int n_paths in
    let variance = Array.fold_left (fun acc x ->
      acc +. (x -. mean) *. (x -. mean)
    ) 0. terminal_wealth /. float_of_int (n_paths - 1) in
    let vol = sqrt variance in
    
    let rf = 0.02 in  (* Risk-free rate *)
    let sharpe = (mean -. rf) /. vol in
    
    {var = var_95; cvar = cvar_95; volatility = vol; sharpe_ratio = sharpe}

  (* Decompose portfolio risk *)
  let decompose_risk params state strategy =
    (* Get price dynamics *)
    let (mu_f, sigma_f) = FuturesPricing.Pricing.compute_price_dynamics 
      params state 0. in
    
    (* Project risk onto factors *)
    let factors = create_drift_matrix params in
    let systematic = matmul factors (matmul sigma_f strategy) in
    
    (* Compute residual risk *)
    let total_risk = matmul sigma_f strategy in
    let idiosyncratic = sub total_risk systematic in
    
    (* Compute basis risk *)
    let n_contracts = size strategy |> Array.get in
    let basis = zeros [|n_contracts|] in
    for i = 0 to n_contracts-2 do
      for j = i+1 to n_contracts-1 do
        let spread = sub (slice sigma_f [|Index.Slice(Some i, Some(i+1), None)|])
                        (slice sigma_f [|Index.Slice(Some j, Some(j+1), None)|]) in
        let basis_ij = matmul spread strategy in
        add_ basis basis_ij
      done
    done;
    
    {systematic; idiosyncratic; basis}
end

(* Market verification and solution analysis *)
module MarketVerification = struct
  type market_properties = {
    complete: bool;
    spanning_rank: int;
    volatility_rank: int;
    martingale: bool;
  }

  type no_arbitrage_result = {
    no_arbitrage: bool;
    price_consistent: bool;
    martingale_property: bool;
  }

  (* Verify market completeness *)
  let verify_market_completion params state t =
    (* Check spanning condition *)
    let check_spanning () =
      let n_contracts = Array.length params.contracts in
      let sensitivity_matrix = Array.init n_contracts (fun i ->
        FuturesPricing.Pricing.compute_ak 
          params t params.contracts.(i).maturity
      ) |> stack ~dim:0 in
      
      let rank = linalg_matrix_rank sensitivity_matrix |> int_value in
      (rank, rank >= 3)  (* Need at least 3 for complete market *)
    in
    
    (* Check volatility matrix rank *)
    let check_volatility () =
      let vol = create_vol_matrix params state in
      let rank = linalg_matrix_rank vol |> int_value in
      (rank, rank >= 3)
    in
    
    let (spanning_rank, spanning) = check_spanning () in
    let (vol_rank, vol_complete) = check_volatility () in
    
    {
      complete = spanning && vol_complete;
      spanning_rank;
      volatility_rank = vol_rank;
      martingale = true;  (* Will be updated by verify_martingale *)
    }

  (* Verify martingale property *)
  let verify_martingale params state t t_end =
    let n_paths = 1000 in
    let dt = 0.01 in
    let steps = int_of_float ((t_end -. t) /. dt) in
    
    (* Simulate price paths *)
    let simulate_path () =
      let rec step state_t prices_t i =
        if i >= steps then prices_t
        else
          let dw = MCTOU_Extension.simulate_brownian_increments params dt `Q in
          let state_next = MCTOU.evolve_q state_t params dt dw in
          
          (* Update futures prices *)
          let new_prices = Array.map (fun contract ->
            FuturesPricing.Pricing.compute_price
              params contract state_next (t +. float_of_int i *. dt)
          ) params.contracts in
          
          step state_next new_prices (i + 1)
      in
      
      (* Initial prices *)
      let initial_prices = Array.map (fun contract ->
        FuturesPricing.Pricing.compute_price params contract state t
      ) params.contracts in
      
      step state initial_prices 0
    in
    
    (* Generate paths *)
    let paths = Array.init n_paths (fun _ -> simulate_path ()) in
    
    (* Check martingale property for each contract *)
    let n_contracts = Array.length params.contracts in
    Array.init n_contracts (fun i ->
      let initial_price = FuturesPricing.Pricing.compute_price
        params params.contracts.(i) state t in
      
      (* Compute terminal expectation *)
      let sum = ref 0. in
      let sum_sq = ref 0. in
      for p = 0 to n_paths-1 do
        let price = paths.(p).(i) in
        sum := !sum +. price;
        sum_sq := !sum_sq +. price *. price
      done;
      
      let mean = !sum /. float_of_int n_paths in
      let var = !sum_sq /. float_of_int n_paths -. mean *. mean in
      let std_err = sqrt (var /. float_of_int n_paths) in
      
      abs_float (mean -. initial_price) <= 2. *. std_err
    )

  (* Verify no-arbitrage conditions *)
  let verify_no_arbitrage params futures_prices state t =
    (* Check price consistency across maturities *)
    let n = Array.length futures_prices in
    let consistent = ref true in
    
    for i = 0 to n-2 do
      for j = i+1 to n-1 do
        let price_i = futures_prices.(i) in
        let price_j = futures_prices.(j) in
        
        (* Check no-arbitrage relation *)
        let drift = create_drift_vector params state in
        let vol = create_vol_matrix params state in
        
        (* Verify price consistency *)
        let ti = params.contracts.(i).maturity in
        let tj = params.contracts.(j).maturity in
        let ak_i = FuturesPricing.Pricing.compute_ak params t ti in
        let ak_j = FuturesPricing.Pricing.compute_ak params t tj in
        
        let ratio = log (price_j /. price_i) in
        let expected_ratio = dot (sub ak_j ak_i) drift |> float_value in
        
        consistent := !consistent && 
                     abs_float (ratio -. expected_ratio) < 1e-6
      done
    done;
    
    (* Check martingale property under Q *)
    let martingale = Array.for_all (fun price ->
      let dw = MCTOU_Extension.simulate_brownian_increments params 0.01 `Q in
      let new_state = MCTOU.evolve_q state params 0.01 dw in
      let dt = 0.01 in
      
      (* Compute expected price change *)
      let (mu_f, sigma_f) = FuturesPricing.Pricing.compute_price_dynamics 
        params state t in
      let drift = dot mu_f (scalar_tensor dt) |> float_value in
      
      abs_float drift < 1e-6
    ) futures_prices in
    
    {
      no_arbitrage = !consistent && martingale;
      price_consistent = !consistent;
      martingale_property = martingale;
    }
end

(* Solution analysis *)
module SolutionAnalysis = struct
  type solution_properties = {
    optimal: bool;
    stable: bool;
    error_bound: float;
    condition_number: float;
  }

  (* Analyze numerical properties *)
  let analyze_numerical_properties params state t solution =
    (* Compute condition number *)
    let compute_condition_number mat =
      let (s, _, _) = svd mat in
      let s_max = maximum s |> float_value in
      let s_min = minimum s |> float_value in
      s_max /. (max s_min 1e-10)
    in
    
    (* Check stability *)
    let check_stability () =
      let dt = 0.01 in
      let v1 = HJB.Solver.solve params state t (t+.dt) solution in
      let v2 = HJB.Solver.solve params state t (t+.2.*.dt) solution in
      
      let diff = norm (sub v2 v1) |> float_value in
      diff < 1e-6
    in
    
    (* Estimate error bound *)
    let estimate_error () =
      let n_grid = 100 in
      let v_fine = HJB.Solver.solve params state t (t+.1.) solution in
      let v_coarse = HJB.Solver.solve ~n_grid params state t (t+.1.) solution in
      
      norm (sub v_fine v_coarse) |> float_value
    in
    
    (* Check optimality *)
    let verify_optimality () =
      let strategy = Portfolio.compute_optimal_strategy params state t in
      let perturbed = add strategy (mul strategy (scalar_tensor 0.01)) in
      
      let v_opt = HJB.Value_Function.compute_value_function params state t 0. in
      let v_pert = HJB.Value_Function.compute_value_function_with_strategy 
        params state t 0. perturbed in
      
      v_opt >= v_pert
    in
    
    (* Get volatility matrix for condition number *)
    let (_, sigma_f) = FuturesPricing.Pricing.compute_price_dynamics 
      params state t in
    let cond_num = compute_condition_number sigma_f in
    
    {
      optimal = verify_optimality ();
      stable = check_stability ();
      error_bound = estimate_error ();
      condition_number = cond_num;
    }

  (* Analyze solution sensitivity *)
  let analyze_sensitivity params state t solution =
    (* Parameter perturbations *)
    let perturbation = 0.01 in
    
    (* Sensitivity to volatility parameters *)
    let vol_sensitivity = 
      let params_up = {params with sigma1 = params.sigma1 *. (1. +. perturbation)} in
      let params_down = {params with sigma1 = params.sigma1 *. (1. -. perturbation)} in
      
      let v_up = HJB.Solver.solve params_up state t (t+.1.) solution in
      let v_down = HJB.Solver.solve params_down state t (t+.1.) solution in
      
      div (sub v_up v_down) (scalar_tensor (2. *. perturbation))
    in
    
    (* Sensitivity to correlation parameters *)
    let corr_sensitivity =
      let params_up = {params with rho12 = params.rho12 +. perturbation} in
      let params_down = {params with rho12 = params.rho12 -. perturbation} in
      
      let v_up = HJB.Solver.solve params_up state t (t+.1.) solution in
      let v_down = HJB.Solver.solve params_down state t (t+.1.) solution in
      
      div (sub v_up v_down) (scalar_tensor (2. *. perturbation))
    in
    
    (* Sensitivity to mean reversion parameters *)
    let mr_sensitivity =
      let params_up = {params with kappa = params.kappa *. (1. +. perturbation)} in
      let params_down = {params with kappa = params.kappa *. (1. -. perturbation)} in
      
      let v_up = HJB.Solver.solve params_up state t (t+.1.) solution in
      let v_down = HJB.Solver.solve params_down state t (t+.1.) solution in
      
      div (sub v_up v_down) (scalar_tensor (2. *. perturbation))
    in
    
    (vol_sensitivity, corr_sensitivity, mr_sensitivity)
end