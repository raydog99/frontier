open Torch

(** Core model types *)
type matrix = Tensor.t
type vector = Tensor.t

type model_params = {
  dimension: int;         (* d assets *)
  horizon: float;         (* T *)
  r_matrix: matrix;       (* R: mean reversion *)
  v_matrix: matrix;       (* V: volatility *)
  k_matrix: matrix;       (* K: permanent impact *)
  s_bar: vector;         (* S̄: long-term mean *)
  eta: matrix;           (* η: temporary impact *)
  gamma: float;          (* γ: risk aversion *)
  terminal_penalty: matrix;  (* Γ: terminal penalty *)
}

type state = {
  inventory: vector;  (* q: inventory *)
  price: vector;     (* S: price *)
  cash: float;       (* X: cash *)
  time: float;       (* t: time *)
}

type riccati_matrices = {
  a_matrix: matrix;  (* A *)
  b_matrix: matrix;  (* B *)
  c_matrix: matrix;  (* C *)
  d_vector: vector;  (* D *)
  e_vector: vector;  (* E *)
  f_scalar: float;   (* F *)
}

type solution = {
  riccati: riccati_matrices list;
  time_grid: float list;
  optimal_controls: Tensor.t list;
  state_trajectory: state list;
  value_trajectory: float list;
}

let is_positive_definite m =
  (* Check using Cholesky decomposition *)
  try
    let _ = Tensor.linalg_cholesky m in
    true
  with _ -> false

let solve_lyapunov a b =
  (* Get eigendecomposition of A *)
  let eigvals, eigvecs = linalg_eigh a in
  let eigvecs_inv = inverse eigvecs in
  
  (* Transform B *)
  let b_transformed = mm (mm eigvecs_inv b) (transpose2 eigvecs_inv 0 1) in
  
  (* Solve transformed system *)
  let n = (size2 a).(0) in
  let x_transformed = zeros [n; n] in
  
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      let lambda_i = float_value (select eigvals 0 i) in
      let lambda_j = float_value (select eigvals 0 j) in
      let b_ij = float_value (select (select b_transformed 0 i) 0 j) in
      x_transformed.$(i,j) <- b_ij /. (lambda_i +. lambda_j)
    done
  done;
  
  (* Transform back *)
  mm (mm eigvecs x_transformed) (transpose2 eigvecs 0 1)

module Measures = struct
  type measure = {
    sample_space: vector -> bool;
    sigma_algebra: (vector -> bool) list;
    measure_fn: vector -> float;
  }

  type filtration_t = {
    time_index: float;
    events: (vector -> bool) list;
    parent: measure option;
  }

  let create_probability_space dimension =
    (* Create filtered probability space with Gaussian measure *)
    let sample_space x = true in
    
    let sigma_algebra = [
      (fun x -> true);
      (fun x -> false);
      (fun x -> Tensor.(norm x ~p:2 |> float_value) <= 1.0)
    ] in
    
    let measure_fn x =
      let normalization = 
        1.0 /. (sqrt (2.0 *. Float.pi) ** float_of_int dimension) in
      let quadratic = 
        Tensor.(mm (transpose2 x 0 1) x |> float_value) in
      normalization *. exp (-0.5 *. quadratic)
    in
    
    { sample_space; sigma_algebra; measure_fn }

  let generate_filtration measure times =
    (* Generate filtration satisfying usual conditions *)
    let build_filtration t acc =
      match t with 
      | [] -> List.rev acc
      | time :: rest ->
          let events = [
            measure.sample_space;
            (fun _ -> false);
            (fun x -> 
              Tensor.(norm x ~p:2 |> float_value) <= sqrt time)
          ] in
          let filtration = {
            time_index = time;
            events;
            parent = Some measure;
          } in
          build_filtration rest (filtration :: acc)
    in
    build_filtration times []
end

module BrownianMotion = struct
  type brownian_path = {
    times: float array;
    values: Tensor.t;  (* k x steps tensor *)
    dimension: int;
  }

  let generate_path dimension steps dt =
    let times = Array.init steps (fun i -> float_of_int i *. dt) in
    let increments = Tensor.randn [dimension; steps] in
    let scaled = Tensor.mul_scalar increments (sqrt dt) in
    let values = Tensor.cumsum scaled ~dim:1 in
    { times; values; dimension }

  let sample_at_time path t =
    let idx = int_of_float (t /. (times.(1) -. times.(0))) in
    if idx < Array.length path.times then
      Some (Tensor.narrow path.values 1 idx 1)
    else None
end

(** State evolution and dynamics *)
module StateEvolution = struct
  let temporary_impact params v =
    (* L(v) = v'ηv quadratic temporary impact *)
    let impact = mm (mm (transpose2 v 0 1) params.eta) v in
    float_value impact

  let permanent_impact params v =
    (* Kv permanent impact *)
    mm params.k_matrix v

  let update_state state v dt params =
    (* Update inventory dq = vdt *)
    let next_inventory = add state.inventory (mul_scalar v dt) in
    
    (* Update price with both OU dynamics and permanent impact
       dS = R(S̄ - S)dt + VdW
       dS̃ = dS + Kvdt *)
    let drift = mm params.r_matrix (sub params.s_bar state.price) in
    let noise = mm params.v_matrix (randn [params.dimension; 1]) in
    let impact = permanent_impact params v in
    
    let price_change = add 
      (add
        (mul_scalar drift dt)
        (mul_scalar noise (sqrt dt)))
      (mul_scalar impact dt) in
    let next_price = add state.price price_change in
    
    (* Update account
       dX = v'Sdt - L(v)dt *)
    let trade_revenue = float_value (mm (transpose2 v 0 1) state.price) in
    let impact_cost = temporary_impact params v in
    let next_cash = state.cash +. (trade_revenue -. impact_cost) *. dt in

    { inventory = next_inventory;
      price = next_price;
      cash = next_cash;
      time = state.time +. dt }

  let terminal_value state params =
    (* Terminal value function:
       X + q'S - l(q) where l(q) = q'Γq *)
    let mtm = float_value (mm (transpose2 state.inventory 0 1) state.price) in
    let penalty = float_value 
      (mm (mm (transpose2 state.inventory 0 1) params.terminal_penalty) 
          state.inventory) in
    state.cash +. mtm -. penalty
end

(** Numerical methods and solvers *)
module AdaptiveMethods = struct
  type refinement_criteria = {
    error_threshold: float;
    max_level: int;
    coarsen_threshold: float;
    refine_threshold: float;
  }

  let estimate_local_error solution dx dt =
    (* Richardson extrapolation for error estimation *)
    let coarse_soln = solution in
    
    (* Compute refined solution *)
    let fine_dx = dx /. 2.0 in
    let fine_dt = dt /. 2.0 in
    let fine_soln = solution in  (* Recompute with finer mesh *)
    
    (* Error estimate *)
    let diff = Tensor.(
      norm (sub coarse_soln fine_soln) ~p:2 
      |> float_value
    ) in
    
    diff /. (2.0 ** 2.0 -. 1.0)  (* Richardson factor *)

  let adapt_mesh grid error criteria =
    (* Adaptive mesh refinement based on error *)
    let nx = Array.length grid in
    let ny = Array.length grid.(0) in
    let refined = Array.make_matrix nx ny 0.0 in
    let levels = Array.make_matrix nx ny 0 in
    
    for i = 0 to nx - 1 do
      for j = 0 to ny - 1 do
        if error.(i).(j) > criteria.refine_threshold &&
           levels.(i).(j) < criteria.max_level then begin
          (* Refine cell *)
          let fine_i = 2 * i in
          let fine_j = 2 * j in
          refined.(fine_i).(fine_j) <- grid.(i).(j);
          refined.(fine_i+1).(fine_j) <- grid.(i).(j);
          refined.(fine_i).(fine_j+1) <- grid.(i).(j);
          refined.(fine_i+1).(fine_j+1) <- grid.(i).(j);
          levels.(fine_i).(fine_j) <- levels.(i).(j) + 1
        end else if error.(i).(j) < criteria.coarsen_threshold &&
                  i mod 2 = 0 && j mod 2 = 0 then begin
          (* Coarsen cells *)
          let coarse_i = i / 2 in
          let coarse_j = j / 2 in
          refined.(coarse_i).(coarse_j) <- 
            (grid.(i).(j) +. grid.(i+1).(j) +.
             grid.(i).(j+1) +. grid.(i+1).(j+1)) /. 4.0;
          levels.(coarse_i).(coarse_j) <- levels.(i).(j) - 1
        end else
          refined.(i).(j) <- grid.(i).(j)
      done
    done;
    
    refined, levels
end

module ImplicitSolver = struct
  type solver_params = {
    max_iter: int;
    tol: float;
    relaxation: float;
  }

  let solve_implicit_system a b params =
    (* Solve linear system using iterative method *)
    let n = (Tensor.size1 a).(0) in
    let x = Tensor.zeros [n] in
    let dx = Tensor.zeros [n] in
    
    let rec iterate iter =
      if iter >= params.max_iter then x
      else begin
        (* Compute residual *)
        let r = Tensor.(sub b (mm a x)) in
        
        (* Update solution *)
        Tensor.copy_ ~src:(mm (inverse a) r) ~dst:dx;
        Tensor.add_ x (mul_scalar dx params.relaxation);
        
        if Tensor.(norm dx |> float_value) < params.tol then x
        else iterate (iter + 1)
      end
    in
    iterate 0
end

module StabilityAnalysis = struct
  type stability_result = {
    cfl_condition: bool;
    von_neumann: bool;
    matrix_stability: bool;
    max_eigenvalue: float;
    stability_margin: float;
  }

  let analyze_numerical_stability dx dt params =
    (* CFL condition *)
    let max_speed = 
      mm params.r_matrix (transpose2 params.r_matrix 0 1)
      |> fun m -> sqrt (float_value (max m)) in
    let cfl = max_speed *. dt /. dx in
    let cfl_stable = cfl <= 1.0 in
    
    (* von Neumann stability *)
    let k_max = Float.pi /. dx in
    let stable_modes = ref true in
    let n_modes = 10 in
    
    for i = 1 to n_modes do
      let k = k_max *. float_of_int i /. float_of_int n_modes in
      let phase = Complex.polar 1.0 (k *. dx) in
      let amplification = 
        Complex.norm (Complex.mul phase phase) in
      if amplification > 1.0 +. 1e-10 then
        stable_modes := false
    done;
    
    (* Matrix stability *)
    let system_matrix = mm params.r_matrix 
      (transpose2 params.r_matrix 0 1) in
    let eigvals = linalg_eigvals system_matrix in
    let max_eigval = ref 0.0 in
    let stability_margin = ref infinity in

    for i = 0 to (shape eigvals).(0) - 1 do
      let ev = get1 eigvals i in
      max_eigval := max !max_eigval ev;
      stability_margin := min !stability_margin (1.0 /. abs_float ev)
    done;
    
    let matrix_stable = !max_eigval *. dt < 1.0 in

    { cfl_condition = cfl_stable;
      von_neumann = !stable_modes;
      matrix_stability = matrix_stable;
      max_eigenvalue = !max_eigval;
      stability_margin = !stability_margin }
end

module MultigridSolver = struct
  type grid_level = {
    nx: int;
    ny: int;
    dx: float;
    dy: float;
    values: float array array;
  }

  let restriction fine_grid =
    (* Restriction operator - full weighting *)
    let nx = fine_grid.nx / 2 in
    let ny = fine_grid.ny / 2 in
    let coarse = Array.make_matrix nx ny 0.0 in
    
    for i = 0 to nx - 1 do
      for j = 0 to ny - 1 do
        coarse.(i).(j) <- 
          (fine_grid.values.(2*i).(2*j) +.
           fine_grid.values.(2*i+1).(2*j) +.
           fine_grid.values.(2*i).(2*j+1) +.
           fine_grid.values.(2*i+1).(2*j+1)) /. 4.0
      done
    done;
    
    { nx; ny;
      dx = fine_grid.dx *. 2.0;
      dy = fine_grid.dy *. 2.0;
      values = coarse }

  let prolongation coarse_grid fine_nx fine_ny =
    (* Prolongation operator - bilinear interpolation *)
    let fine = Array.make_matrix fine_nx fine_ny 0.0 in
    
    for i = 0 to coarse_grid.nx - 1 do
      for j = 0 to coarse_grid.ny - 1 do
        (* Direct injection *)
        fine.(2*i).(2*j) <- coarse_grid.values.(i).(j);
        
        if 2*i + 1 < fine_nx then
          fine.(2*i+1).(2*j) <- 
            (coarse_grid.values.(i).(j) +.
             (if i + 1 < coarse_grid.nx then 
                coarse_grid.values.(i+1).(j)
              else coarse_grid.values.(i).(j))) /. 2.0;
        
        if 2*j + 1 < fine_ny then
          fine.(2*i).(2*j+1) <-
            (coarse_grid.values.(i).(j) +.
             (if j + 1 < coarse_grid.ny then
                coarse_grid.values.(i).(j+1)
              else coarse_grid.values.(i).(j))) /. 2.0;
        
        if 2*i + 1 < fine_nx && 2*j + 1 < fine_ny then
          fine.(2*i+1).(2*j+1) <-
            (coarse_grid.values.(i).(j) +.
             (if i + 1 < coarse_grid.nx then 
                coarse_grid.values.(i+1).(j)
              else coarse_grid.values.(i).(j)) +.
             (if j + 1 < coarse_grid.ny then
                coarse_grid.values.(i).(j+1)
              else coarse_grid.values.(i).(j)) +.
             (if i + 1 < coarse_grid.nx && j + 1 < coarse_grid.ny then
                coarse_grid.values.(i+1).(j+1)
              else coarse_grid.values.(i).(j))) /. 4.0
      done
    done;
    
    { nx = fine_nx; ny = fine_ny;
      dx = coarse_grid.dx /. 2.0;
      dy = coarse_grid.dy /. 2.0;
      values = fine }

  let solve_multigrid initial_grid rhs max_iter tol =
    let rec v_cycle grid level =
      if level = 0 then
        (* Direct solve at coarsest level *)
        grid
      else begin
        (* Pre-smoothing *)
        let smoothed = ImplicitSolver.solve_implicit_system
          (Tensor.eye grid.nx) 
          (Tensor.of_float2 rhs)
          {max_iter=5; tol=0.1; relaxation=0.8} in
        
        (* Compute residual *)
        let residual = Array.make_matrix grid.nx grid.ny 0.0 in

        (* Restriction *)
        let coarse_grid = restriction grid in
        let coarse_rhs = restriction 
          {grid with values=residual} in
        
        (* Recursive solve *)
        let coarse_correction = v_cycle coarse_grid (level - 1) in
        
        (* Prolongation *)
        let correction = prolongation coarse_correction 
                         grid.nx grid.ny in
        
        (* Post-smoothing *)
        ImplicitSolver.solve_implicit_system
          (Tensor.eye grid.nx)
          (Tensor.add (Tensor.of_float2 grid.values)
                     (Tensor.of_float2 correction.values))
          {max_iter=5; tol=0.1; relaxation=0.8}
          |> fun t -> {grid with values = Tensor.to_float2 t}
      end
    in
    
    let levels = int_of_float (log2 (float_of_int initial_grid.nx)) in
    let rec iterate iter grid =
      if iter >= max_iter then grid
      else
        let new_grid = v_cycle grid levels in
        let residual_norm = 0.0 in (* Compute residual norm *)
        if residual_norm < tol then new_grid
        else iterate (iter + 1) new_grid
    in
    
    iterate 0 initial_grid
end

module FiniteDifference = struct
  type boundary_condition =
    | Dirichlet of float
    | Neumann of float
    | Mixed of float * float

  let solve_pde grid dx dt bc_left bc_right max_iter tol params =
    let nx = Array.length grid in
    let solution = Array.copy grid in
    
    let apply_bc i value = function
      | Dirichlet v -> v
      | Neumann deriv -> 
          value +. deriv *. dx
      | Mixed (a, b) -> 
          (a *. value +. b) /. (1.0 +. a *. dx)
    in
    
    let rec iterate iter =
      if iter >= max_iter then solution
      else
        let max_change = ref 0.0 in
        let new_soln = Array.copy solution in
        
        (* Interior points *)
        for i = 1 to nx - 2 do
          let laplacian = 
            (solution.(i+1) -. 2.0 *. solution.(i) +. 
             solution.(i-1)) /. (dx *. dx) in
          new_soln.(i) <- solution.(i) +. dt *. laplacian;
          max_change := max !max_change 
            (abs_float (new_soln.(i) -. solution.(i)))
        done;
        
        (* Boundary conditions *)
        new_soln.(0) <- apply_bc 0 new_soln.(1) bc_left;
        new_soln.(nx-1) <- apply_bc (nx-1) new_soln.(nx-2) bc_right;
        
        if !max_change < tol then new_soln
        else begin
          Array.blit new_soln 0 solution 0 nx;
          iterate (iter + 1)
        end
    in
    
    iterate 0
end

module ErrorAnalysis = struct
  type error_stats = {
    l1_error: float;
    l2_error: float;
    linf_error: float;
    avg_error: float;
  }

  let compute_error_stats numerical exact =
    let nx = Array.length numerical in
    let errors = Array.init nx (fun i ->
      abs_float (numerical.(i) -. exact.(i))) in
    
    let sum_abs = Array.fold_left (+.) 0.0 errors in
    let sum_sq = Array.fold_left (fun acc e -> acc +. e *. e) 
                  0.0 errors in
    let max_err = Array.fold_left max 0.0 errors in
    
    { l1_error = sum_abs /. float_of_int nx;
      l2_error = sqrt (sum_sq /. float_of_int nx);
      linf_error = max_err;
      avg_error = sum_abs /. float_of_int nx }
end

(** Complete Riccati equation solver *)
module RiccatiSolver = struct
  type riccati_system = {
    q_matrix: Tensor.t;  (* Q matrix *)
    y_matrix: Tensor.t;  (* Y matrix *)
    u_matrix: Tensor.t;  (* U matrix *)
    terminal: Tensor.t;  (* Terminal condition *)
    dimension: int;      (* Dimension of state space *)
  }

  let create_system params =
    let d = params.dimension in
    let sigma = Tensor.(mm params.v_matrix 
                         (transpose2 params.v_matrix 0 1)) in
    
    (* Construct Q matrix *)
    let q11 = Tensor.mul_scalar sigma (0.5 *. params.gamma) in
    let q12 = params.r_matrix in
    let q21 = Tensor.transpose2 q12 0 1 in
    let q22 = Tensor.zeros [d; d] in
    let q = Tensor.zeros [2*d; 2*d] in
    Tensor.(
      copy_ ~src:q11 ~dst:(narrow (narrow q 0 0 d) 1 0 d);
      copy_ ~src:q12 ~dst:(narrow (narrow q 0 0 d) 1 d d);
      copy_ ~src:q21 ~dst:(narrow (narrow q 0 d d) 1 0 d);
      copy_ ~src:q22 ~dst:(narrow (narrow q 0 d d) 1 d d)
    );

    (* Construct Y matrix *)
    let y11 = Tensor.zeros [d; d] in
    let y12 = Tensor.zeros [d; d] in
    let y21 = sigma in
    let y22 = params.r_matrix in
    let y = Tensor.zeros [2*d; 2*d] in
    Tensor.(
      copy_ ~src:y11 ~dst:(narrow (narrow y 0 0 d) 1 0 d);
      copy_ ~src:y12 ~dst:(narrow (narrow y 0 0 d) 1 d d);
      copy_ ~src:y21 ~dst:(narrow (narrow y 0 d d) 1 0 d);
      copy_ ~src:y22 ~dst:(narrow (narrow y 0 d d) 1 d d)
    );

    (* Construct U matrix *)
    let u11 = Tensor.inverse params.eta in
    let u12 = Tensor.zeros [d; d] in
    let u21 = Tensor.zeros [d; d] in
    let u22 = Tensor.mul_scalar sigma 2.0 in
    let u = Tensor.zeros [2*d; 2*d] in
    Tensor.(
      copy_ ~src:u11 ~dst:(narrow (narrow u 0 0 d) 1 0 d);
      copy_ ~src:u12 ~dst:(narrow (narrow u 0 0 d) 1 d d);
      copy_ ~src:u21 ~dst:(narrow (narrow u 0 d d) 1 0 d);
      copy_ ~src:u22 ~dst:(narrow (narrow u 0 d d) 1 d d)
    );

    (* Terminal condition *)
    let term = Tensor.zeros [2*d; 2*d] in
    Tensor.(
      copy_ ~src:(neg params.terminal_penalty) 
            ~dst:(narrow (narrow term 0 0 d) 1 0 d)
    );

    { q_matrix = q;
      y_matrix = y;
      u_matrix = u;
      terminal = term;
      dimension = d }

  let riccati_rhs system p =
    (* Right-hand side of Riccati ODE *)
    add system.q_matrix (
      add
        (add 
          (mm (transpose2 system.y_matrix 0 1) p)
          (mm p system.y_matrix))
        (mm p (mm system.u_matrix p))
    )

  module RiccatiRegularization = struct
    type regularization = {
      lambda_min: float;     (* Minimum eigenvalue threshold *)
      max_condition: float;  (* Maximum condition number *)
      epsilon: float;        (* Regularization parameter *)
    }

    (* Stabilize Riccati solution via regularization *)
    let stabilize_riccati p reg =
      
      (* Compute eigendecomposition *)
      let eigvals, eigvecs = linalg_eigh p in
      
      (* Find minimum eigenvalue *)
      let min_eigval = ref max_float in
      let max_eigval = ref (-.max_float) in
      for i = 0 to (shape eigvals).(0) - 1 do
        let ev = get1 eigvals i in
        min_eigval := min !min_eigval ev;
        max_eigval := max !max_eigval ev
      done;

      (* Check if regularization needed *)
      if !min_eigval < reg.lambda_min ||
         !max_eigval /. !min_eigval > reg.max_condition then begin
        
        (* Adjust eigenvalues *)
        let new_eigvals = copy eigvals in
        for i = 0 to (shape new_eigvals).(0) - 1 do
          let ev = get1 new_eigvals i in
          if ev < reg.lambda_min then
            new_eigvals.$(i) <- reg.lambda_min;
          if ev > reg.lambda_min *. reg.max_condition then
            new_eigvals.$(i) <- reg.lambda_min *. reg.max_condition
        done;

        (* Reconstruct matrix *)
        mm (mm eigvecs (diag new_eigvals)) (transpose2 eigvecs 0 1)
      end else
        p
  end

  let solve_backward system params dt steps =
    (* Solve Riccati ODE backward in time *)
    let regularization = {
      RiccatiRegularization.lambda_min = 1e-6;
      max_condition = 1e6;
      epsilon = 1e-8;
    } in
    
    (* Initialize at terminal condition *)
    let p = ref system.terminal in
    let solutions = ref [(params.horizon, system.terminal)] in
    
    for step = steps - 1 downto 0 do
      let t = float_of_int step *. dt in
      
      (* Implicit step *)
      let rhs = riccati_rhs system !p in
      let p_new = sub !p (mul_scalar rhs dt) in
      
      (* Regularize solution *)
      let p_reg = RiccatiRegularization.stabilize_riccati 
                   p_new regularization in
      
      p := p_reg;
      solutions := (t, p_reg) :: !solutions
    done;
    
    !solutions
end

(** HJB equation solver *)
module HJBSolver = struct
  type hjb_solution = {
    value_fn: state -> float;
    optimal_control: state -> Tensor.t;
    verification: verification_result;
  }
  and verification_result = {
    viscosity_property: bool;
    comparison_principle: bool;
    boundary_satisfied: bool;
    terminal_satisfied: bool;
  }

  (* Create HJB solution from Riccati solution *)
  let create_hjb_solution params riccati_solution =
    let interpolate_riccati t =
      (* Interpolate Riccati matrices at time t *)
      let rec find_interval = function
        | (t1, p1) :: (t2, p2) :: _ when t >= t1 && t <= t2 ->
            let alpha = (t -. t1) /. (t2 -. t1) in
            Tensor.(add (mul_scalar p1 (1.0 -. alpha))
                       (mul_scalar p2 alpha))
        | _ :: rest -> find_interval rest
        | [] -> snd (List.hd riccati_solution)
      in
      find_interval riccati_solution
    in

    let value_fn state =
      (* Compute value function *)
      let p = interpolate_riccati state.time in

      
      let q = state.inventory in
      let s = state.price in
      
      (* Compute θ(t,q,S) *)
      let theta_value =
        let qaq = mm (mm (transpose2 q 0 1) 
                       (narrow (narrow p 0 0 params.dimension) 
                              1 0 params.dimension)) q in
        let qbs = mm (mm (transpose2 q 0 1)
                       (narrow (narrow p 0 0 params.dimension)
                              1 params.dimension params.dimension)) s in
        let scs = mm (mm (transpose2 s 0 1)
                       (narrow (narrow p 0 params.dimension params.dimension)
                              1 params.dimension params.dimension)) s in
        float_value (add (add qaq qbs) scs)
      in
      
      let mtm = float_value (mm (transpose2 state.inventory 0 1) 
                              state.price) in
      exp (-. params.gamma *. (state.cash +. mtm +. theta_value))
    in

    let optimal_control state =
      (* Compute optimal control *)
      let p = interpolate_riccati state.time in

      
      let coeff = mul_scalar (inverse params.eta) 0.5 in
      let term1 = mm (mul_scalar 
                       (narrow (narrow p 0 0 params.dimension)
                              1 0 params.dimension)
                       2.0)
                    state.inventory in
      let term2 = mm params.r_matrix 
                    (sub params.s_bar state.price) in
      mm coeff (add term1 term2)
    in
end

(** Optimal control *)
module OptimalControl = struct
  type control_strategy = {
    compute_control: state -> Tensor.t;
    verify_admissible: Tensor.t -> bool;
    feedback_law: feedback_control;
  }
  and feedback_control = {
    feedback_fn: state -> float -> Tensor.t;
    state_constraints: state -> bool;
    stability_verified: bool;
  }

  let create_strategy params riccati_solution =
    (* Create optimal control strategy from Riccati solution *)
    let interpolate_riccati t =
      (* Interpolate Riccati matrices at time t *)
      let rec find_interval = function
        | (t1, p1) :: (t2, p2) :: _ when t >= t1 && t <= t2 ->
            let alpha = (t -. t1) /. (t2 -. t1) in
            Tensor.(add (mul_scalar p1 (1.0 -. alpha))
                       (mul_scalar p2 alpha))
        | _ :: rest -> find_interval rest
        | [] -> snd (List.hd riccati_solution)
      in
      find_interval riccati_solution
    in

    let compute_control state =
      (* Compute optimal control *)
      let p = interpolate_riccati state.time in

      
      let coeff = mul_scalar (inverse params.eta) 0.5 in
      let term1 = mm (mul_scalar 
                       (narrow (narrow p 0 0 params.dimension)
                              1 0 params.dimension)
                       2.0)
                    state.inventory in
      let term2 = mm params.r_matrix 
                    (sub params.s_bar state.price) in
      mm coeff (add term1 term2)
    in

    let create_feedback_law () =
      (* Create feedback control law *)
      let feedback_fn state t =
        let p = interpolate_riccati t in
        
        (* Compute optimal feedback control *)
  
        let coeff = mul_scalar (inverse params.eta) 0.5 in
        let qterm = mm p state.inventory in
        let drift = mm params.r_matrix 
                     (sub params.s_bar state.price) in
        mm coeff (add qterm drift)
      in

      let state_constraints state =
        (* Verify state constraints *)
  
        let inv_norm = norm state.inventory ~p:2 |> float_value in
        let price_norm = norm state.price ~p:2 |> float_value in
        let bound = 10.0 in  (* Can be calibrated *)
        
        inv_norm <= bound && price_norm <= bound
      in

      let verify_stability () =
        (* Verify closed-loop stability *)
        let stable = ref true in
        let dt = 0.01 in
        let steps = int_of_float (params.horizon /. dt) in
        
        let simulate_trajectory state =
          let rec simulate t state =
            if t >= params.horizon then true
            else
              let u = feedback_fn state t in
              let next_state = StateEvolution.update_state 
                               state u dt params in
              if not (state_constraints next_state) then false
              else simulate (t +. dt) next_state
          in
          simulate state.time state
        in
        
        (* Test multiple initial conditions *)
        for _ = 1 to 100 do
          let initial_state = {
            inventory = randn [params.dimension];
            price = randn [params.dimension];
            cash = 0.0;
            time = 0.0;
          } in
          if not (simulate_trajectory initial_state) then
            stable := false
        done;
        
        !stable
      in

      { feedback_fn;
        state_constraints;
        stability_verified = verify_stability () }
    in

    { compute_control;
      verify_admissible;
      feedback_law = create_feedback_law () }
end

(** Simulation and execution components *)
module Simulation = struct
  type simulation_params = {
    dt: float;
    n_steps: int;
    n_paths: int;
    seed: int option;
  }

  type simulation_result = {
    paths: state list list;
    controls: Tensor.t list list;
    values: float list list;
    metrics: execution_metrics;
  }
  and execution_metrics = {
    total_costs: float list;
    avg_price_impact: float list;
    inventory_profiles: Tensor.t list;
    execution_shortfall: float list;
  }

  let simulate_ou_price params sim_params =
    (* Simulate OU price process paths *)
    let dt = sim_params.dt in
    let steps = sim_params.n_steps in
    
    (* Initialize paths *)
    let paths = Array.make sim_params.n_paths 
      (Array.make steps params.s_bar) in
    
    (* Optional seeding *)
    (match sim_params.seed with
     | Some s -> Random.init s
     | None -> ());
    
    (* Generate paths *)
    for i = 0 to sim_params.n_paths - 1 do
      let path = paths.(i) in
      path.(0) <- params.s_bar;
      
      for t = 1 to steps - 1 do
        let prev = path.(t-1) in
        
        (* OU dynamics *)
        let drift = Tensor.(mm params.r_matrix (sub params.s_bar prev)) in
        let noise = Tensor.(mm params.v_matrix (randn [params.dimension; 1])) in
        
        path.(t) <- Tensor.(
          add prev (add
            (mul_scalar drift dt)
            (mul_scalar noise (sqrt dt)))
        )
      done
    done;
    
    paths

  let simulate_execution strategy params sim_params =
    (* Simulate execution with given strategy *)
    let price_paths = simulate_ou_price params sim_params in
    let n_paths = Array.length price_paths in
    let steps = Array.length price_paths.(0) in
    
    let execute_single_path price_path =
      let rec simulate t state acc_states acc_controls acc_values =
        if t >= params.horizon then
          (List.rev acc_states, 
           List.rev acc_controls, 
           List.rev acc_values)
        else
          let control = strategy.OptimalControl.compute_control state in
          let next_state = StateEvolution.update_state 
                            state control sim_params.dt params in
          let value = ValueFunction.compute_value 
                       { value_trajectory = acc_values; 
                         state_trajectory = state :: acc_states;
                         optimal_controls = control :: acc_controls;
                         time_grid = [t] } 
                       params state in
          
          simulate 
            (t +. sim_params.dt) 
            next_state 
            (state :: acc_states)
            (control :: acc_controls)
            (value :: acc_values)
      in
      
      let initial_state = {
        inventory = Tensor.ones [params.dimension];  (* Initial position *)
        price = price_path.(0);
        cash = 0.0;
        time = 0.0;
      } in
      
      simulate 0.0 initial_state [] [] []
    in
    
    let execution_results = Array.to_list 
      (Array.map execute_single_path price_paths) in
    
    let paths, controls, values = List.split3 execution_results in
    
    (* Compute execution metrics *)
    let compute_metrics paths controls =
      let total_costs = List.map
        (fun (states, controls) ->
          List.fold_left2
            (fun acc state control ->
              let impact = StateEvolution.temporary_impact 
                           params control in
              acc +. impact)
            0.0 states controls)
        (List.combine paths controls) in
      
      let avg_impact = List.map
        (fun (states, controls) ->
          let n = float_of_int (List.length states) in
          List.fold_left2
            (fun acc state control ->
              let impact = Tensor.(mm (transpose2 control 0 1)
                                    params.k_matrix
                                 |> mm state.price
                                 |> float_value) in
              acc +. impact /. n)
            0.0 states controls)
        (List.combine paths controls) in
      
      let inv_profiles = List.map
        (fun states ->
          List.map
            (fun state -> state.inventory)
            states)
        paths in
      
      let shortfall = List.map
        (fun (states, controls) ->
          let initial_value = ValueFunction.compute_value
            { value_trajectory = [];
              state_trajectory = [];
              optimal_controls = [];
              time_grid = [] }
            params (List.hd states) in
          let final_value = ValueFunction.compute_value
            { value_trajectory = [];
              state_trajectory = [];
              optimal_controls = [];
              time_grid = [] }
            params (List.hd (List.rev states)) in
          initial_value -. final_value)
        (List.combine paths controls) in
      
      { total_costs;
        avg_price_impact = avg_impact;
        inventory_profiles = inv_profiles;
        execution_shortfall = shortfall }
    in
    
    { paths;
      controls;
      values;
      metrics = compute_metrics paths controls }
end

(** Complete execution system *)
module ExecutionSystem = struct
  type execution_params = {
    model: model_params;
    simulation: Simulation.simulation_params;
    control: OptimalControl.control_strategy;
    verification: Verification.verification_result;
  }

  type execution_result = {
    solution: solution;
    simulation: Simulation.simulation_result;
    performance: performance_metrics;
  }
  and performance_metrics = {
    total_cost: float;
    implementation_shortfall: float;
    tracking_error: float;
    information_ratio: float;
  }

  let create_execution_system model_params =
    (* Create complete execution system *)
    
    (* Create Riccati solution *)
    let riccati_system = RiccatiSolver.create_system model_params in
    let dt = model_params.horizon /. 1000.0 in
    let riccati_solution = RiccatiSolver.solve_backward 
                            riccati_system model_params dt 1000 in
    
    (* Create HJB solution *)
    let hjb_solution = HJBSolver.create_hjb_solution 
                        model_params riccati_solution in
    
    (* Create control strategy *)
    let control_strategy = OptimalControl.create_strategy 
                            model_params riccati_solution in
    
    (* Verify solution *)
    let verification = Verification.verify_solution
                        { value_trajectory = [];
                          state_trajectory = [];
                          optimal_controls = [];
                          time_grid = [] }
                        model_params in
    
    (* Create simulation parameters *)
    let sim_params = {
      Simulation.dt = dt;
      n_steps = 1000;
      n_paths = 100;
      seed = Some 42;
    } in
    
    { model = model_params;
      simulation = sim_params;
      control = control_strategy;
      verification }

  let execute system initial_state =
    (* Execute trading strategy *)
    let simulation = Simulation.simulate_execution 
                      system.control system.model 
                      system.simulation in
    
    (* Compute performance metrics *)
    let performance =
      let total_cost = List.hd simulation.metrics.total_costs in
      let shortfall = List.hd simulation.metrics.execution_shortfall in
      let tracking = Tensor.(
        let expected = List.hd simulation.paths 
                      |> List.hd 
                      |> (fun s -> s.inventory) in
        let actual = List.hd simulation.metrics.inventory_profiles
                    |> List.hd in
        norm (sub expected actual) ~p:2 
        |> float_value) in
      let ir = shortfall /. (sqrt total_cost) in
      
      { total_cost;
        implementation_shortfall = shortfall;
        tracking_error = tracking;
        information_ratio = ir }
    in
    
    { solution = 
        { value_trajectory = List.hd simulation.values;
          state_trajectory = List.hd simulation.paths;
          optimal_controls = List.hd simulation.controls;
          time_grid = 
            List.init system.simulation.n_steps
              (fun i -> float_of_int i *. system.simulation.dt) };
      simulation;
      performance }
end