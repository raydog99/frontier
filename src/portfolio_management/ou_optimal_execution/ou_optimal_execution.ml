open Torch

type execution_params = {
    time_horizon: float;
    risk_aversion: float;
    price_sensitivity: Tensor.t;
    transaction_cost: Tensor.t;
    terminal_penalty: Tensor.t;
}

type market_params = {
    mean_reversion: Tensor.t;  (* R matrix *)
    volatility: Tensor.t;      (* V matrix *)
    long_term_mean: Tensor.t;  (* S_bar *)
    initial_price: Tensor.t;   (* S_0 *)
}

type state = {
    time: float;
    inventory: Tensor.t;
    price: Tensor.t;
    cash: float;
}

(* Core tensor operations *)
module Tensor = struct
  let trace m = 
    let dims = size m in
    if List.length dims < 2 then failwith "Matrix required for trace"
    else begin
      let d = List.hd dims in
      diagonal m ~dim1:0 ~dim2:1 |> sum
    end

  let quadratic_form x a y =
    mm (mm (transpose x ~dim0:0 ~dim1:1) a) y 
    |> reshape [] 
    |> float_value

  let symmetric_part m =
    let mt = transpose m ~dim0:0 ~dim1:1 in
    div (add m mt) (scalar_tensor 2.0)
end

(* Probability space and filtration *)
module Filtration = struct
  type time = float

  type filtration = {
    start_time: time;
    end_time: time;
    time_steps: int;
    dt: float;
  }

  let make_filtration ~start_time ~end_time ~time_steps =
    let dt = (end_time -. start_time) /. float_of_int time_steps in
    {start_time; end_time; time_steps; dt}

  let time_points filtration =
    let rec generate_points t points remaining =
      if remaining <= 0 then List.rev points
      else generate_points (t +. filtration.dt) (t :: points) (remaining - 1)
    in
    generate_points filtration.start_time [] (filtration.time_steps + 1)
end

(* Stochastic process *)
module Process = struct
  type 'a t = {
    value: 'a;
    time: float;
    filtration: Filtration.filtration;
  }

  let create value time filtration = {value; time; filtration}

  (* Brownian motion increment *)
  let brownian_increment ~dim dt =
    let scale = sqrt dt in
    mul (randn [dim]) scale

  (* Generate correlated Brownian paths *)
  let generate_correlated_brownian filt dim correlation_matrix =
    let cholesky = cholesky correlation_matrix in
    let times = Filtration.time_points filt in
    let increments = List.map (fun _ -> brownian_increment ~dim filt.dt) times in
    let correlated_increments = 
      List.map (fun incr -> mm cholesky incr) increments in
    List.fold_left (fun acc incr -> add acc incr) (zeros [dim]) correlated_increments
end

(* Inventory process *)
module Inventory = struct
  type t = {
    quantity: Tensor.t;
    trading_rate: Tensor.t -> Tensor.t;  (* Feedback control function *)
    process: Tensor.t Process.t;
  }

  let evolve state dt =
    let dq = mul state.trading_rate state.process.value dt in
    add state.quantity dq

  let create ~initial_quantity ~trading_rate ~filtration =
    {
      quantity = initial_quantity;
      trading_rate;
      process = Process.create initial_quantity filtration.Filtration.start_time filtration;
    }
end

(* Price process *)
module Price = struct
  type price_dynamics = {
    mean_reversion: Tensor.t;       (* R matrix *)
    volatility: Tensor.t;           (* V matrix *)
    long_term_mean: Tensor.t;       (* S̄ *)
    correlation: Tensor.t;          (* Correlation matrix *)
  }

  type t = {
    fundamental: Tensor.t Process.t;
    market: Tensor.t Process.t;
    dynamics: price_dynamics;
    price_sensitivity: Tensor.t;     (* K matrix *)
  }

  (* Fundamental price evolution *)
  let evolve_fundamental state dt =
    let drift = mul (mm state.dynamics.mean_reversion 
                   (sub state.dynamics.long_term_mean state.fundamental.value)) dt in
    let diffusion = mm state.dynamics.volatility 
                   (Process.brownian_increment ~dim:(size state.fundamental.value |> List.hd) dt) in
    add state.fundamental.value (add drift diffusion)

  (* Market price evolution *)
  let evolve_market state trading_rate dt =
    let fundamental_change = sub (evolve_fundamental state dt) state.fundamental.value in
    let impact = mul (mm state.price_sensitivity trading_rate) dt in
    add state.market.value (add fundamental_change impact)

  let create ~initial_price ~dynamics ~price_sensitivity ~filtration =
    {
      fundamental = Process.create initial_price filtration.Filtration.start_time filtration;
      market = Process.create initial_price filtration.Filtration.start_time filtration;
      dynamics;
      price_sensitivity;
    }

  (* Compute covariance matrix *)
  let compute_covariance dynamics =
    mm dynamics.volatility (transpose dynamics.volatility ~dim0:0 ~dim1:1)

  (* Compute drift *)
  let compute_drift dynamics price =
    mm dynamics.mean_reversion (sub dynamics.long_term_mean price)
end

(* Limit order book *)
module LimitOrderBook = struct
  type t = {
    balance: float;
    process: float Process.t;
    transaction_cost: Tensor.t;     (* η matrix *)
  }

  (* Order book account evolution *)
  let evolve state trading_rate price dt =
    let execution_cost = float_value (
      mm (mm (transpose trading_rate ~dim0:0 ~dim1:1) state.transaction_cost) trading_rate
    ) in
    let trading_pnl = float_value (
      mul (mm (transpose trading_rate ~dim0:0 ~dim1:1) price) dt
    ) in
    state.balance +. trading_pnl -. execution_cost

  let create ~initial_balance ~transaction_cost ~filtration =
    {
      balance = initial_balance;
      process = Process.create initial_balance filtration.Filtration.start_time filtration;
      transaction_cost;
    }
end

(* Utility function *)
module UtilityFunction = struct
  type value_params = {
    a: Tensor.t;  (* Matrix A(t) *)
    b: Tensor.t;  (* Matrix B(t) *)
    c: Tensor.t;  (* Matrix C(t) *)
    d: Tensor.t;  (* Vector D(t) *)
    e: Tensor.t;  (* Vector E(t) *)
    f: float;     (* Scalar F(t) *)
  }

  (* Theta function *)
  let theta params state =
    let q = state.inventory in
    let s = state.price in
    
    let term1 = Tensor.quadratic_form q params.a q in
    let term2 = Tensor.quadratic_form q params.b s in
    let term3 = Tensor.quadratic_form s params.c s in
    let term4 = float_value (mm (transpose params.d ~dim0:0 ~dim1:1) q) in
    let term5 = float_value (mm (transpose params.e ~dim0:0 ~dim1:1) s) in
    
    term1 +. term2 +. term3 +. term4 +. term5 +. params.f

  (* Utility function *)
  let utility params state =
    let wealth = state.cash +. 
      Tensor.quadratic_form state.inventory (eye (List.hd (size state.inventory))) state.price in
    -. exp (-. params.risk_aversion *. (wealth +. theta params state))

  (* Compute gradient *)
  let gradient utility_func state =
    let q = state.inventory in
    let s = state.price in
    grad utility_func [q; s]

  (* Compute Hessian *)
  let hessian utility_func state =
    let q = state.inventory in
    let s = state.price in
    hessian utility_func [q; s]
end

(* HJB *)
module HJB = struct
  type hjb_params = {
    execution: execution_params;
    market: market_params;
  }

  (* Hamiltonian computation *)
  let hamiltonian params state control value_grad =
    let covar = Price.compute_covariance params.market in
    
    (* Trading costs *)
    let execution_cost = 
      Tensor.quadratic_form control params.execution.transaction_cost control in
    
    (* Price impact *)
    let price_impact = 
      Tensor.quadratic_form control params.execution.price_sensitivity state.price in
    
    (* Mean reversion term *)
    let mean_rev = Price.compute_drift params.market state.price in
    
    (* Drift term *)
    let drift = add mean_rev (mul control params.execution.price_sensitivity) in
    
    (* Value function gradient terms *)
    let value_term = Tensor.quadratic_form value_grad covar value_grad in
    
    drift, execution_cost, value_term

  (* Optimal control computation *)
  let optimal_control params state value_grad =
    let temp_inv = inverse params.execution.transaction_cost in
    let control = mm temp_inv value_grad in
    neg (div control (scalar_tensor 2.0))

  (* HJB right-hand side *)
  let hjb_rhs params state value value_grad value_hess =
    let drift, exec_cost, value_term = 
      hamiltonian params state (optimal_control params state value_grad) value_grad in
    
    let diffusion_term = Tensor.trace (mm value_hess 
      (Price.compute_covariance params.market)) in
    
    add drift (add (neg exec_cost) (add value_term diffusion_term))

  (* Compute diffusion term *)
  let compute_diffusion_term utility_func state =
    let grad = UtilityFunction.gradient utility_func state in
    let hess = UtilityFunction.hessian utility_func state in
    mm hess grad
end

(* Matrix Riccati solver *)
module Riccati = struct
  type riccati_params = {
    q: Tensor.t;  (* Q matrix *)
    y: Tensor.t;  (* Y matrix *)
    u: Tensor.t;  (* U matrix *)
  }

  let make_params q y u = {q; y; u}

  (* P' = Q + Y'P + PY + PUP  *)
  let riccati_derivative params p =
    let term1 = params.q in
    let term2 = mm (transpose params.y ~dim0:0 ~dim1:1) p in
    let term3 = mm p params.y in
    let term4 = mm (mm p params.u) p in
    add (add (add term1 term2) term3) term4

  (* Newton iteration for implicit step *)
  let newton_step params p_curr p_prev dt =
    let f = sub p_curr (add p_prev (mul (riccati_derivative params p_curr) dt)) in
    let dp = solve_linear_system (jacobian (riccati_derivative params) p_curr) f in
    sub p_curr dp

  (* Solve Riccati over time interval with implicit method *)
  let solve params p0 tstart tend dt =
    let steps = int_of_float ((tend -. tstart) /. dt) in
    let max_newton_iter = 10 in
    let newton_tol = 1e-6 in
    
    let rec newton_iterate p_prev p_curr iter =
      if iter >= max_newton_iter then p_curr
      else 
        let p_next = newton_step params p_curr p_prev dt in
        if float_value (norm (sub p_next p_curr)) < newton_tol then p_next
        else newton_iterate p_prev p_next (iter + 1)
    in
    
    let rec solve_steps p step =
      if step >= steps then p
      else 
        let p_guess = add p (mul (riccati_derivative params p) dt) in
        let p_next = newton_iterate p p_guess 0 in
        solve_steps p_next (step + 1)
    in
    solve_steps p0 0
end

(* System solver for full ODE system *)
module SystemSolver = struct
  type system_state = {
    a: Tensor.t;
    b: Tensor.t;
    c: Tensor.t;
    d: Tensor.t;
    e: Tensor.t;
    f: float;
  }

  let make_state ~a ~b ~c ~d ~e ~f = {a; b; c; d; e; f}

  (* ODE system *)
  let derivative params state =
    let sigma = mm params.market.volatility 
                  (transpose params.market.volatility ~dim0:0 ~dim1:1) in
    
    (* A' *)
    let a_dot = 
      let term1 = mul (add state.b (eye (List.hd (size state.b)))) 
                     (mm sigma (add (transpose state.b ~dim0:0 ~dim1:1) 
                                  (eye (List.hd (size state.b))))) in
      let term2 = mm (mm state.a params.execution.transaction_cost) state.a in
      sub (mul term1 params.execution.risk_aversion) term2 in
    
    (* B' *)
    let b_dot =
      let term1 = mm (add state.b (eye (List.hd (size state.b)))) 
                     params.market.mean_reversion in
      let term2 = mul (mm (add state.b (eye (List.hd (size state.b)))) 
                         (mm sigma state.c)) 
                     (scalar_tensor 2.0) in
      let term3 = mm (mm state.a params.execution.transaction_cost) state.b in
      sub (add term1 term2) term3 in
      
    (* C' *)
    let c_dot = 
      let term1 = add (mm (transpose params.market.mean_reversion ~dim0:0 ~dim1:1) state.c)
                     (mm state.c params.market.mean_reversion) in
      let term2 = mul (mm (mm state.c sigma) state.c) (scalar_tensor 2.0) in
      let term3 = mul (mm (transpose state.b ~dim0:0 ~dim1:1) 
                        (mm params.execution.transaction_cost state.b))
                     (scalar_tensor 0.25) in
      add (add term1 term2) term3 in

    (* D' *)
    let d_dot =
      let term1 = mm (add state.b (eye (List.hd (size state.b)))) 
                     (mm params.market.mean_reversion params.market.long_term_mean) in
      let term2 = mm (add state.b (eye (List.hd (size state.b))))
                     (mm sigma state.e) in
      let term3 = mm (mm state.a params.execution.transaction_cost) state.d in
      sub (add term1 term2) term3 in

    (* E' *)
    let e_dot =
      let term1 = mul (mm state.c 
                        (mm params.market.mean_reversion params.market.long_term_mean))
                     (scalar_tensor 2.0) in
      let term2 = mm (transpose params.market.mean_reversion ~dim0:0 ~dim1:1) state.e in
      let term3 = mul (mm (mm state.c sigma) state.e) (scalar_tensor 2.0) in
      let term4 = mul (mm (transpose state.b ~dim0:0 ~dim1:1) 
                        (mm params.execution.transaction_cost state.d))
                     (scalar_tensor 0.5) in
      add (add (add term1 term2) term3) term4 in

    (* F' *)
    let f_dot =
      let term1 = float_value (sum (mm (transpose params.market.long_term_mean ~dim0:0 ~dim1:1)
                                     (mm (transpose params.market.mean_reversion ~dim0:0 ~dim1:1) 
                                         state.e))) in
      let term2 = -.float_value (Tensor.trace (mm sigma state.c)) in
      let term3 = params.execution.risk_aversion *. 
                 (float_value (sum (mm (transpose state.e ~dim0:0 ~dim1:1)
                                     (mm sigma state.e)))) /. 2.0 in
      let term4 = -.float_value (sum (mm (transpose state.d ~dim0:0 ~dim1:1)
                                       (mm params.execution.transaction_cost state.d))) /. 4.0 in
      term1 +. term2 +. term3 +. term4 in

    make_state ~a:a_dot ~b:b_dot ~c:c_dot ~d:d_dot ~e:e_dot ~f:f_dot

  (* Forward Euler method *)
  let solve params initial_state t_start t_end dt =
    let steps = int_of_float ((t_end -. t_start) /. dt) in
    let rec solve_steps state step =
      if step >= steps then state
      else
        let deriv = derivative params state in
        let next_state = {
          a = add state.a (mul deriv.a dt);
          b = add state.b (mul deriv.b dt);
          c = add state.c (mul deriv.c dt);
          d = add state.d (mul deriv.d dt);
          e = add state.e (mul deriv.e dt);
          f = state.f +. deriv.f *. dt;
        } in
        solve_steps next_state (step + 1)
    in
    solve_steps initial_state 0
end

(* Numerical methods and utilities *)
module Numerical = struct
  (* Matrix condition number *)
  let condition_number m =
    let s = svd m in
    let max_sv = max s in
    let min_sv = min s in
    div max_sv min_sv

  (* Check numerical stability *)
  let is_numerically_stable m tol =
    float_value (condition_number m) < tol

  (* Stable matrix inverse *)
  let stable_inverse m tol =
    if is_numerically_stable m tol then
      Some (inverse m)
    else None

  (* Regularized inverse for ill-conditioned matrices *)
  let regularized_inverse m lambda =
    let n = List.hd (size m) in
    let reg = mul (eye n) (scalar_tensor lambda) in
    inverse (add m reg)

  (* Adaptive step size control *)
  type step_control = {
    dt: float;
    error_est: float;
    accept: bool;
  }

  let adaptive_step ~dt ~error ~tol ~factor =
    let new_dt = if error > tol then dt *. factor else dt /. factor in
    {dt = new_dt; error_est = error; accept = error <= tol}
end

(* Optimal execution *)
module OptimalExecution = struct
  type execution_state = {
    time: float;
    inventory: Inventory.t;
    price: Price.t;
    cash: LimitOrderBook.t;
    utility_function: UtilityFunction.value_params;
  }

  (* Optimal control computation *)
  let optimal_control params state =
    let grad_theta = add 
      (mm state.utility_function.a state.inventory.quantity)
      (mm state.utility_function.b state.price.market.value) in
    neg (div (mm (inverse params.execution.transaction_cost) grad_theta) 
            (scalar_tensor 2.0))

  (* Evolution *)
  let evolution params state dt =
    let control = optimal_control params state in
    {
      time = state.time +. dt;
      inventory = { state.inventory with
        quantity = Inventory.evolve state.inventory dt };
      price = { state.price with
        market = { state.price.market with
          value = Price.evolve_market state.price control dt }};
      cash = { state.cash with
        balance = LimitOrderBook.evolve state.cash control state.price.market.value dt };
      utility_function = state.utility_function;
    }

  (* Solve optimal execution problem *)
  let solve params initial_state =
    let dt = params.filtration.dt in
    let riccati_params = {
      Riccati.
      q = params.dynamics.mean_reversion;
      y = params.dynamics.volatility;
      u = params.execution.transaction_cost;
    } in
    
    (* Solve Riccati for value function coefficients *)
    let d = List.hd (size params.dynamics.mean_reversion) in
    let p0 = zeros [2 * d; 2 * d] in
    let p = Riccati.solve riccati_params p0 0.0 params.time_horizon dt in
    
    (* Extract value function coefficients *)
    let a = narrow p ~dim:0 ~start:0 ~length:d |>
            fun x -> narrow x ~dim:1 ~start:0 ~length:d in
    let b = narrow p ~dim:0 ~start:0 ~length:d |>
            fun x -> narrow x ~dim:1 ~start:d ~length:d in
    let c = narrow p ~dim:0 ~start:d ~length:d |>
            fun x -> narrow x ~dim:1 ~start:d ~length:d in
    
    let value_params = UtilityFunction.{
      a = a;
      b = b;
      c = c;
      d = zeros [d];
      e = zeros [d];
      f = 0.0;
    } in
    
    (* Time stepping *)
    let rec step state =
      if state.time >= params.time_horizon then state
      else
        let next_state = evolution params state dt in
        step next_state
    in
    
    let initial_execution_state = {
      time = 0.0;
      inventory = initial_state.inventory;
      price = initial_state.price;
      cash = initial_state.cash;
      utility_function = value_params;
    } in
    
    step initial_execution_state
end

(* Solution interface *)
module Solution = struct
  type error_components = {
    local_error: float;
    global_error: float;
    stability_factor: float;
  }

  type solution_result = {
    final_state: OptimalExecution.execution_state;
    verification: Verification.verification_result;
    error_analysis: error_components;
  }

  (* Main solver *)
  let solve params initial_state =
    (* Solve optimal execution problem *)
    let final_state = OptimalExecution.solve params initial_state in
    
    (* Verify solution *)
    let verification = Verification.verify_hjb 
      params final_state.utility_function final_state params.filtration.dt in
    
    (* Compute error estimates *)
    let error_analysis = {
      local_error = verification.error_bound;
      global_error = verification.error_bound *. 
        float_of_int (int_of_float (params.time_horizon /. params.filtration.dt));
      stability_factor = List.assoc "condition_number" verification.stability_metrics;
    } in
    
    if verification.is_valid then
      Ok {final_state; verification; error_analysis}
    else
      Error "Solution verification failed"

  (* Create solver parameters *)
  let make_params ~time_horizon ~risk_aversion ~mean_reversion ~volatility 
                  ~long_term_mean ~initial_price ~price_sensitivity ~transaction_cost
                  ~terminal_penalty ~time_steps =
    let filtration = Filtration.make_filtration 
      ~start_time:0.0 ~end_time:time_horizon ~time_steps in
    {
      execution = {
        time_horizon;
        risk_aversion;
        price_sensitivity;
        transaction_cost;
        terminal_penalty;
      };
      market = {
        mean_reversion;
        volatility;
        long_term_mean;
        initial_price;
      };
      filtration;
    }

  (* Create initial state *)
  let make_initial_state ~initial_inventory ~initial_price ~initial_cash ~filtration =
    {
      inventory = Inventory.create 
        ~initial_quantity:initial_inventory
        ~trading_rate:(fun _ -> zeros_like initial_inventory)
        ~filtration;
      price = Price.create
        ~initial_price
        ~dynamics:{
          mean_reversion = zeros [1; 1];
          volatility = zeros [1; 1];
          long_term_mean = initial_price;
          correlation = eye (List.hd (size initial_price));
        }
        ~price_sensitivity:(zeros [1; 1])
        ~filtration;
      cash = LimitOrderBook.create
        ~initial_balance:initial_cash
        ~transaction_cost:(zeros [1; 1])
        ~filtration;
    }
end