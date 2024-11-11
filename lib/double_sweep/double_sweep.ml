module Types = struct
  type grid_type = 
    | Uniform
    | Hyperbolic of float
    | Square_root

  type grid_params = {
    n_time: int;
    n_space: int;
    x_min: float;
    x_max: float;
    t_max: float;
    grid_type: grid_type;
  }

  type market_params = {
    r: float;
    mu: float;
    sigma: float;
    s0: float;
    k: float;
    is_time_dependent: bool;
  }

  type solve_status = {
    converged: bool;
    iterations: int;
    error: float;
  }
end

module BlackScholes = struct
  open Types
  open Torch

  let make_grid params grid_type =
    match grid_type with
    | Uniform -> 
        let dx = (params.x_max -. params.x_min) /. float_of_int (params.n_space - 1) in
        let dt = params.t_max /. float_of_int (params.n_time - 1) in
        let x = Tensor.linspace ~start:params.x_min ~end_:params.x_max ~steps:params.n_space in
        let t = Tensor.linspace ~start:0. ~end_:params.t_max ~steps:params.n_time in
        (x, t, dx, dt)
    
    | Hyperbolic alpha ->
        let xi = Tensor.linspace ~start:(-1.) ~end_:1. ~steps:params.n_space in
        let x = Tensor.mul_scalar (Tensor.tanh (Tensor.mul_scalar xi alpha)) 
                 (params.x_max -. params.x_min) in
        let dt = params.t_max /. float_of_int (params.n_time - 1) in
        let t = Tensor.linspace ~start:0. ~end_:params.t_max ~steps:params.n_time in
        let dx = (params.x_max -. params.x_min) /. float_of_int (params.n_space - 1) in
        (x, t, dx, dt)
    
    | Square_root ->
        let x = Tensor.linspace ~start:params.x_min ~end_:params.x_max ~steps:params.n_space in
        let t_sqrt = Tensor.linspace ~start:0. ~end_:(sqrt params.t_max) ~steps:params.n_time in
        let t = Tensor.pow t_sqrt 2.0 in
        let dx = (params.x_max -. params.x_min) /. float_of_int (params.n_space - 1) in
        let dt = params.t_max /. float_of_int (params.n_time - 1) in
        (x, t, dx, dt)

  let build_coefficients grid_params market_params x dt =
    let n = grid_params.n_space in
    let dx = (grid_params.x_max -. grid_params.x_min) /. float_of_int (n - 1) in
    
    let a = Tensor.zeros [n] in
    let b = Tensor.zeros [n] in
    let c = Tensor.zeros [n] in
    
    (* Interior points *)
    for i = 1 to n-2 do
      let x_i = Tensor.get x i |> Tensor.to_float0_exn in
      let sigma2 = market_params.sigma *. market_params.sigma in
      
      (* Il'in scheme coefficient *)
      let pe = market_params.mu *. dx /. sigma2 in
      let w = if abs_float pe < 1e-10 then 1.0
              else pe /. (exp pe -. 1.0) in
      
      let ai = -.(w *. market_params.mu *. x_i +. sigma2 *. x_i *. x_i /. dx) in
      let ci = w *. market_params.mu *. x_i -. sigma2 *. x_i *. x_i /. dx in
      let bi = -.ai -. ci +. market_params.r *. dx in
      
      Tensor.fill_float1_idx a i (ai *. dt);
      Tensor.fill_float1_idx b i (1.0 +. bi *. dt);
      Tensor.fill_float1_idx c i (ci *. dt)
    done;
    
    (* Boundary conditions *)
    Tensor.fill_float1_idx a 0 0.0;
    Tensor.fill_float1_idx b 0 (1.0 +. market_params.r *. dt);
    Tensor.fill_float1_idx c 0 0.0;
    
    Tensor.fill_float1_idx a (n-1) 0.0;
    Tensor.fill_float1_idx b (n-1) (1.0 +. market_params.r *. dt);
    Tensor.fill_float1_idx c (n-1) 0.0;
    
    (a, b, c)

  let initial_conditions grid_params market_params x is_put =
    let payoff = 
      if is_put then
        Tensor.max (Tensor.sub market_params.k x) (Tensor.zeros_like x)
      else
        Tensor.max (Tensor.sub x market_params.k) (Tensor.zeros_like x)
    in
    payoff

  let apply_boundary_conditions v x t market_params is_put =
    let n = Tensor.size x 0 in
    let modified = Tensor.copy v in
    
    if is_put then begin
      (* Put option boundaries *)
      Tensor.fill_float1_idx modified 0 
        (market_params.k *. exp (-. market_params.r *. (market_params.t_max -. t)));
      Tensor.fill_float1_idx modified (n-1) 0.0
    end else begin
      (* Call option boundaries *)
      Tensor.fill_float1_idx modified 0 0.0;
      let x_max = Tensor.get x (n-1) |> Tensor.to_float0_exn in
      Tensor.fill_float1_idx modified (n-1)
        (x_max -. market_params.k *. exp (-. market_params.r *. (market_params.t_max -. t)))
    end;
    
    modified
end

module ErrorAnalysis = struct
  open Types
  open Torch

  type error_component = {
    truncation: float;
    roundoff: float;
    iteration: float;
    boundary: float;
  }

  type stability_metric = {
    spectral_radius: float;
    condition_number: float;
    growth_factor: float;
    dissipation_error: float;
    dispersion_error: float;
  }

  let analyze_truncation_error solution dt dx =
    let n = Tensor.size solution 0 in
    let errors = Tensor.zeros [n] in
    
    (* Richardson extrapolation *)
    for i = 1 to n-2 do
      let dt_half = dt /. 2.0 in
      let val_i = Tensor.get solution i |> Tensor.to_float0_exn in
      
      let val_half = ref val_i in
      for _ = 1 to 2 do
        val_half := !val_half +. dt_half *. (!val_half -. val_i) /. 2.0
      done;
      
      let error = abs_float (val_i -. !val_half) /. 3.0 in
      Tensor.fill_float1_idx errors i error
    done;
    
    errors

  let analyze_stability a b c dt =
    let n = Tensor.size a 0 in
    
    (* Power iteration for spectral radius *)
    let v = Tensor.ones [n] in
    let lambda = ref 0.0 in
    let iter = ref 0 in
    let max_iter = 1000 in
    let tol = 1e-12 in
    
    while !iter < max_iter do
      let av = Tensor.zeros [n] in
      for i = 0 to n-1 do
        let sum = ref 0.0 in
        if i > 0 then
          sum := !sum +. (Tensor.get a i |> Tensor.to_float0_exn) *. 
                       (Tensor.get v (i-1) |> Tensor.to_float0_exn);
        sum := !sum +. (Tensor.get b i |> Tensor.to_float0_exn) *. 
                     (Tensor.get v i |> Tensor.to_float0_exn);
        if i < n-1 then
          sum := !sum +. (Tensor.get c i |> Tensor.to_float0_exn) *. 
                       (Tensor.get v (i+1) |> Tensor.to_float0_exn);
        Tensor.fill_float1_idx av i !sum
      done;
      
      let norm = Tensor.norm ~p:2 av |> Tensor.to_float0_exn in
      lambda := norm;
      Tensor.div_scalar_ av norm;
      Tensor.copy_ v av;
      
      incr iter
    done;
    
    let spectral_radius = !lambda in
    let condition_number = spectral_radius /. 
      (Tensor.min (Tensor.abs b) |> Tensor.to_float0_exn) in
    
    {
      spectral_radius;
      condition_number;
      growth_factor = spectral_radius;
      dissipation_error = abs_float (1.0 -. spectral_radius);
      dispersion_error = 0.0;  
    }

  let estimate_total_error solution dt dx stability =
    let trunc = analyze_truncation_error solution dt dx in
    let weight = 1.0 /. (1.0 +. stability.condition_number) in
    Tensor.mul_scalar trunc weight
end

module TRBDF2 = struct
  open Types
  open Torch

  let alpha = 2.0 -. sqrt 2.0

  let trapezoidal_stage current l_operator dt =
    let f_star = ref (Tensor.copy current) in
    let converged = ref false in
    let iter = ref 0 in
    let max_iter = 100 in
    let tol = 1e-12 in
    
    while not !converged && !iter < max_iter do
      let prev = Tensor.copy !f_star in
      
      let l_fn = l_operator current in
      let l_fstar = l_operator !f_star in
      
      let rhs = Tensor.add current (Tensor.mul_scalar 
        (Tensor.add l_fn l_fstar) (alpha *. dt /. 2.0)) in
      f_star := rhs;
      
      let diff = Tensor.sub !f_star prev in
      let error = Tensor.norm ~p:Float.infinity diff 
                 |> Tensor.to_float0_exn in
      converged := error < tol;
      incr iter
    done;
    
    !f_star

  let bdf2_stage f_star current l_operator dt =
    let f_next = ref (Tensor.copy f_star) in
    let converged = ref false in
    let iter = ref 0 in
    let max_iter = 100 in
    let tol = 1e-12 in
    
    while not !converged && !iter < max_iter do
      let prev = Tensor.copy !f_next in
      
      let l_fnext = l_operator !f_next in
      
      let term1 = Tensor.mul_scalar f_star (1.0 /. alpha) in
      let term2 = Tensor.mul_scalar current 
        (-.(1.0 -. alpha) ** 2.0 /. alpha) in
      let term3 = Tensor.mul_scalar l_fnext 
        ((1.0 -. alpha) *. dt) in
      
      let sum = Tensor.add (Tensor.add term1 term2) term3 in
      f_next := Tensor.mul_scalar sum (1.0 /. (2.0 -. alpha));
      
      let diff = Tensor.sub !f_next prev in
      let error = Tensor.norm ~p:Float.infinity diff 
                 |> Tensor.to_float0_exn in
      converged := error < tol;
      incr iter
    done;
    
    !f_next

  let step dt current l_operator payoff market_params =
    (* First stage: Trapezoidal *)
    let f_star = trapezoidal_stage current l_operator dt in
    let f_star = Tensor.maximum f_star payoff in
    
    (* Second stage: BDF2 *)
    let f_next = bdf2_stage f_star current l_operator dt in
    let f_next = Tensor.maximum f_next payoff in
    
    (* Error estimation *)
    let error = Tensor.sub f_star f_next in
    let error_norm = Tensor.norm ~p:2 error |> Tensor.to_float0_exn in
    
    (f_next, error_norm)
end


module IlinScheme = struct
  open Types
  open Torch

  type scheme_params = {
    beta: float;
    use_limiting: bool;
    stabilization: [`None | `SUPG | `Full];
  }

  let compute_coefficients dx mu sigma r dt params =
    let n = Array.length dx in
    let a = Array.make n 0.0 in
    let b = Array.make n 0.0 in
    let c = Array.make n 0.0 in
    
    for i = 1 to n-2 do
      let dx_i = dx.(i) in
      let dx_im = dx.(i-1) in
      let dx_half = (dx_im +. dx_i) /. 2.0 in
      let x_i = float_of_int i *. dx_i in
      
      let sigma_i = sigma x_i in
      let mu_i = mu x_i in
      let sigma2 = sigma_i *. sigma_i in
      
      (* Il'in function with artificial diffusion *)
      let pe = mu_i *. dx_half /. sigma2 in
      let w = if abs_float pe < 1e-10 then 1.0
              else begin
                let b = pe *. (1.0 +. params.beta) in
                b /. (exp b -. 1.0)
              end in
      
      (* Add SUPG stabilization if requested *)
      let tau = match params.stabilization with
        | `SUPG -> 
            let h = dx_half in
            let u = abs_float mu_i in
            h /. (2.0 *. u) *. (1.0 -. 1.0 /. (abs_float pe))
        | `Full ->
            let h = dx_half in
            let u = abs_float mu_i in
            h /. (2.0 *. u)
        | `None -> 0.0 in
      
      let ai = -.(w *. mu_i +. sigma2 /. dx_im) in
      let ci = w *. mu_i -. sigma2 /. dx_i in
      let bi = -.ai -. ci +. r *. dx_half in
      
      if params.use_limiting then begin
        let r_minus = if i > 1 then
          let df_minus = (ai -. a.(i-1)) /. dx_im in
          let df_plus = (c.(i) -. ci) /. dx_i in
          if abs_float df_plus > 1e-10 then df_minus /. df_plus else 1.0
        else 1.0 in
        
        let phi = max 0.0 (min 2.0 r_minus) in
        a.(i) <- ai *. phi;
        c.(i) <- ci *. phi
      end else begin
        a.(i) <- ai;
        c.(i) <- ci
      end;
      
      b.(i) <- bi +. (if tau > 0.0 then 2.0 *. tau *. mu_i *. mu_i /. dx_half else 0.0)
    done;
    
    (* Barrier conditions *)
    a.(0) <- 0.0;
    b.(0) <- 1.0 +. dt *. r;
    c.(0) <- 0.0;
    
    a.(n-1) <- 0.0;
    b.(n-1) <- 1.0 +. dt *. r;
    c.(n-1) <- 0.0;
    
    (a, b, c)

  let solve dx mu sigma r dt params initial barrier =
    let (a, b, c) = compute_coefficients dx mu sigma r dt params in
    DoubleSweepLU.solve 
      (Tensor.of_float1 a)
      (Tensor.of_float1 b)
      (Tensor.of_float1 c)
      (Tensor.of_float1 initial)
      (Tensor.of_float1 barrier)
      |> (fun result -> result.solution)
end

module DoubleSweepLU = struct
  open Types
  open Torch

  type sweep_result = {
    solution: Tensor.t;
    barrier_indices: (int * int) option;
    convergence: solve_status;
  }

  let luul_decomposition a b c =
    let n = Tensor.size a 0 in
    let l = Tensor.zeros [n] in
    let u = Tensor.zeros [n] in
    
    (* Forward LU decomposition *)
    let l00 = Tensor.get b 0 |> Tensor.to_float0_exn in
    if abs_float l00 < 1e-14 then
      failwith "Zero pivot encountered in LU decomposition";
    
    Tensor.fill_float1_idx l 0 l00;
    let u01 = (Tensor.get c 0 |> Tensor.to_float0_exn) /. l00 in
    Tensor.fill_float1_idx u 0 u01;
    
    for i = 1 to n-1 do
      let li = Tensor.get a i |> Tensor.to_float0_exn in
      let ui_prev = if i > 1 then 
        Tensor.get u (i-1) |> Tensor.to_float0_exn else u01 in
      let lii = (Tensor.get b i |> Tensor.to_float0_exn) -. li *. ui_prev in
      
      if abs_float lii < 1e-14 then
        failwith (Printf.sprintf "Zero pivot at i=%d" i);
      
      Tensor.fill_float1_idx l i lii;
      
      if i < n-1 then begin
        let uii = (Tensor.get c i |> Tensor.to_float0_exn) /. lii in
        Tensor.fill_float1_idx u i uii
      end
    done;
    
    (l, u)

  let solve_forward l u v =
    let n = Tensor.size v 0 in
    let y = Tensor.zeros [n] in
    let z = Tensor.zeros [n] in
    
    (* Forward substitution *)
    let y0 = (Tensor.get v 0 |> Tensor.to_float0_exn) /.
             (Tensor.get l 0 |> Tensor.to_float0_exn) in
    Tensor.fill_float1_idx y 0 y0;
    
    for i = 1 to n-1 do
      let sum = ref 0.0 in
      if i > 0 then
        sum := !sum +. (Tensor.get l i |> Tensor.to_float0_exn) *.
                     (Tensor.get y (i-1) |> Tensor.to_float0_exn);
      
      let yi = ((Tensor.get v i |> Tensor.to_float0_exn) -. !sum) /.
               (Tensor.get l i |> Tensor.to_float0_exn) in
      Tensor.fill_float1_idx y i yi
    done;
    
    (* Backward substitution *)
    Tensor.fill_float1_idx z (n-1) 
      (max (Tensor.get y (n-1) |> Tensor.to_float0_exn) 0.0);
    
    for i = n-2 downto 0 do
      let sum = ref 0.0 in
      if i < n-1 then
        sum := !sum +. (Tensor.get u i |> Tensor.to_float0_exn) *.
                     (Tensor.get z (i+1) |> Tensor.to_float0_exn);
      
      let zi = max ((Tensor.get y i |> Tensor.to_float0_exn) -. !sum) 0.0 in
      Tensor.fill_float1_idx z i zi
    done;
    
    z

  let solve a b c v payoff =
    let (l, u) = luul_decomposition a b c in
    let solution = solve_forward l u v in
    let solution = Tensor.maximum solution payoff in
    
    (* Find exercise barriers *)
    let n = Tensor.size solution 0 in
    let barriers = ref None in
    let in_exercise = ref false in
    let start_idx = ref 0 in
    
    for i = 0 to n-1 do
      let val_i = Tensor.get solution i |> Tensor.to_float0_exn in
      let payoff_i = Tensor.get payoff i |> Tensor.to_float0_exn in
      
      if abs_float (val_i -. payoff_i) < 1e-10 then begin
        if not !in_exercise then begin
          in_exercise := true;
          start_idx := i
        end
      end else if !in_exercise then begin
        barriers := Some (!start_idx, i-1);
        in_exercise := false
      end
    done;
    
    if !in_exercise then
      barriers := Some (!start_idx, n-1);
    
    {
      solution;
      barrier_indices = !barriers;
      convergence = {
        converged = true;
        iterations = 1;
        error = 0.0;
      }
    }

  let solve_two_barriers a b c v payoff =
    let n = Tensor.size v 0 in
    let (l, u) = luul_decomposition a b c in
    
    (* Forward and backward sweeps *)
    let forward = solve_forward l u v in
    let backward = solve_forward u l v in
    
    (* Combine solutions *)
    let solution = Tensor.zeros [n] in
    for i = 0 to n-1 do
      let f1 = Tensor.get forward i |> Tensor.to_float0_exn in
      let f2 = Tensor.get backward i |> Tensor.to_float0_exn in
      let p = Tensor.get payoff i |> Tensor.to_float0_exn in
      Tensor.fill_float1_idx solution i (max (max f1 f2) p)
    done;
    
    {
      solution;
      barrier_indices = None;
      convergence = {
        converged = true;
        iterations = 1;
        error = 0.0;
      }
    }
end

module MMatrixVerification = struct
  open Types
  open Torch

  type violation_type =
    | DiagonalNonPositive
    | OffDiagonalNonNegative
    | NonDominantDiagonal
    | Reducible
    | BarrierConditionViolation

  type violation = {
    kind: violation_type;
    location: int;
    magnitude: float;
    details: string;
  }

  let verify_m_matrix_properties a b c =
    let n = Tensor.size a 0 in
    let violations = ref [] in
    
    (* Check diagonal positivity *)
    for i = 0 to n-1 do
      let bi = Tensor.get b i |> Tensor.to_float0_exn in
      if bi <= 0.0 then
        violations := {
          kind = DiagonalNonPositive;
          location = i;
          magnitude = -.bi;
          details = "Non-positive diagonal element";
        } :: !violations
    done;
    
    (* Check off-diagonal non-positivity *)
    for i = 1 to n-1 do
      let ai = Tensor.get a i |> Tensor.to_float0_exn in
      if ai > 0.0 then
        violations := {
          kind = OffDiagonalNonNegative;
          location = i;
          magnitude = ai;
          details = "Positive lower diagonal element";
        } :: !violations
    done;
    
    for i = 0 to n-2 do
      let ci = Tensor.get c i |> Tensor.to_float0_exn in
      if ci > 0.0 then
        violations := {
          kind = OffDiagonalNonNegative;
          location = i;
          magnitude = ci;
          details = "Positive upper diagonal element";
        } :: !violations
    done;
    
    (* Check diagonal dominance *)
    for i = 1 to n-2 do
      let ai = abs_float (Tensor.get a i |> Tensor.to_float0_exn) in
      let bi = abs_float (Tensor.get b i |> Tensor.to_float0_exn) in
      let ci = abs_float (Tensor.get c i |> Tensor.to_float0_exn) in
      
      if bi <= ai +. ci then
        violations := {
          kind = NonDominantDiagonal;
          location = i;
          magnitude = (ai +. ci) -. bi;
          details = "Non-dominant diagonal";
        } :: !violations
    done;
    
    !violations

  let verify_barrier_conditions a b c r =
    let n = Tensor.size a 0 in
    let violations = ref [] in
    
    (* Check left barrier condition *)
    let b0 = Tensor.get b 0 |> Tensor.to_float0_exn in
    let c0 = Tensor.get c 0 |> Tensor.to_float0_exn in
    
    if b0 +. c0 < 0.0 || (r < 0.0 && c0 > 0.0) then
      violations := {
        kind = BarrierConditionViolation;
        location = 0;
        magnitude = abs_float (b0 +. c0);
        details = "Left barrier condition violated";
      } :: !violations;
    
    (* Check right barrier condition *)
    let an = Tensor.get a (n-1) |> Tensor.to_float0_exn in
    let bn = Tensor.get b (n-1) |> Tensor.to_float0_exn in
    
    if an +. bn < 0.0 || (r < 0.0 && an > 0.0) then
      violations := {
        kind = BarrierConditionViolation;
        location = n-1;
        magnitude = abs_float (an +. bn);
        details = "Right barrier condition violated";
      } :: !violations;
    
    !violations

  let verify_all a b c r =
    let standard_violations = verify_m_matrix_properties a b c in
    let barrier_violations = verify_barrier_conditions a b c r in
    standard_violations @ barrier_violations
end

module TRBDFEigenAnalysis = struct
  open Types
  open Torch

  type eigenvalue_result = {
    max_real: float;
    max_imag: float;
    condition_number: float;
    stability_region: (float * float) list;
  }

  let power_method a max_iter tol =
    let n = Tensor.size a 0 in
    let v = Tensor.ones [n] in
    let lambda = ref 0.0 in
    let converged = ref false in
    let iter = ref 0 in
    
    while not !converged && !iter < max_iter do
      let v_old = Tensor.copy v in
      
      (* Power iteration *)
      let av = Tensor.matmul a v in
      let norm = Tensor.norm ~p:2 av |> Tensor.to_float0_exn in
      Tensor.div_scalar_ av norm;
      Tensor.copy_ v av;
      
      (* Update eigenvalue estimate *)
      let new_lambda = 
        Tensor.dot (Tensor.matmul a v) v |> Tensor.to_float0_exn in
      
      let diff = abs_float (new_lambda -. !lambda) in
      converged := diff < tol;
      lambda := new_lambda;
      incr iter
    done;
    
    (!lambda, !iter, !converged)

  let compute_eigenspectrum a max_iter tol =
    let n = Tensor.size a 0 in
    let h = Tensor.copy a in
    let eigenvals = ref [] in
    let converged = ref false in
    let iter = ref 0 in
    
    (* QR iteration *)
    while not !converged && !iter < max_iter do
      let q = Tensor.zeros [n; n] in
      let r = Tensor.zeros [n; n] in
      
      (* QR decomposition *)
      for j = 0 to n-1 do
        let v = Tensor.narrow h ~dim:1 ~start:j ~length:1 in
        let norm = Tensor.norm ~p:2 v |> Tensor.to_float0_exn in
        
        for i = 0 to j-1 do
          let qi = Tensor.narrow q ~dim:1 ~start:i ~length:1 in
          let dot = Tensor.dot qi v |> Tensor.to_float0_exn in
          Tensor.fill_float2_idx r i j dot;
          Tensor.add_ v (Tensor.mul_scalar qi (-.dot))
        done;
        
        let norm_j = Tensor.norm ~p:2 v |> Tensor.to_float0_exn in
        if norm_j > tol then begin
          Tensor.div_scalar_ v norm_j;
          Tensor.copy_ (Tensor.narrow q ~dim:1 ~start:j ~length:1) v;
          Tensor.fill_float2_idx r j j norm_j
        end
      done;
      
      (* Update H = RQ *)
      Tensor.matmul_ h r q;
      
      (* Check convergence *)
      let max_subdiag = ref 0.0 in
      for i = 0 to n-2 do
        let val_ = abs_float (Tensor.get h i (i+1) |> Tensor.to_float0_exn) in
        max_subdiag := max !max_subdiag val_
      done;
      
      converged := !max_subdiag < tol;
      incr iter;
      
      if !converged then
        for i = 0 to n-1 do
          eigenvals := (Tensor.get h i i |> Tensor.to_float0_exn) :: !eigenvals
        done
    done;
    
    List.rev !eigenvals

  let solve_trbdf2_stability l_operator dt =
    let eigenvals = compute_eigenspectrum l_operator 1000 1e-12 in
    
    let stability_points = ref [] in
    let max_real = ref (-.infinity) in
    let max_imag = ref (-.infinity) in
    
    List.iter (fun lambda ->
      (* TR stage stability *)
      let z = Complex.{ re = lambda *. dt; im = 0.0 } in
      let alpha = 2.0 -. sqrt 2.0 in
      
      let tr_factor = Complex.div
        (Complex.add Complex.one 
          (Complex.mul (Complex.scalar (alpha/.2.0)) z))
        (Complex.sub Complex.one 
          (Complex.mul (Complex.scalar (alpha/.2.0)) z)) in
      
      (* BDF2 stage stability *)
      let bdf_factor = Complex.div Complex.one
        (Complex.add Complex.one 
          (Complex.mul (Complex.scalar (1.0-.alpha)) z)) in
      
      (* Combined stability factor *)
      let combined = Complex.mul tr_factor bdf_factor in
      
      max_real := max !max_real combined.Complex.re;
      max_imag := max !max_imag combined.Complex.im;
      
      stability_points := (combined.Complex.re, combined.Complex.im) 
                         :: !stability_points
    ) eigenvals;
    
    let condition_number = 
      let sorted = List.sort compare eigenvals in
      let max_ev = List.hd (List.rev sorted) in
      let min_ev = List.hd sorted in
      abs_float (max_ev /. min_ev)
    in
    
    {
      max_real = !max_real;
      max_imag = !max_imag;
      condition_number;
      stability_region = !stability_points;
    }
end

module IntersectionPointHandler = struct
  open Types
  open Torch

  type intersection_detail = {
    time: float;
    location: int * int;
    precise_point: float * float;
    pre_velocities: float * float;
    post_behavior: [`Continuous | `Discontinuous of float];
  }

  let find_intersection_point prev_sol curr_sol prev_t curr_t x =
    let n = Tensor.size x 0 in
    let dx = Tensor.get x 1 |> Tensor.to_float0_exn in
    let dt = curr_t -. prev_t in
    
    let crossings = ref [] in
    let prev_diff = ref 0.0 in
    
    for i = 1 to n-2 do
      let prev_val = Tensor.get prev_sol i |> Tensor.to_float0_exn in
      let curr_val = Tensor.get curr_sol i |> Tensor.to_float0_exn in
      let diff = curr_val -. prev_val in
      
      if !prev_diff *. diff < 0.0 then
        crossings := (i-1, i, !prev_diff, diff) :: !crossings;
      
      prev_diff := diff
    done;
    
    match !crossings with
    | [(i1, i2, d1, d2)] ->
        let x1 = Tensor.get x i1 |> Tensor.to_float0_exn in
        let x2 = Tensor.get x i2 |> Tensor.to_float0_exn in
        
        (* Interpolate intersection point *)
        let alpha = -.d1 /. (d2 -. d1) in
        let x_intersect = x1 +. alpha *. (x2 -. x1) in
        
        (* Compute velocities *)
        let v1 = d1 /. dt in
        let v2 = d2 /. dt in
        
        (* Check post-intersection behavior *)
        let post_behavior =
          let post_d1 = Tensor.get curr_sol i1 |> Tensor.to_float0_exn in
          let post_d2 = Tensor.get curr_sol i2 |> Tensor.to_float0_exn in
          let post_slope = (post_d2 -. post_d1) /. (x2 -. x1) in
          
          if abs_float post_slope > 1e-6 then
            `Discontinuous post_slope
          else
            `Continuous
        in
        
        Some {
          time = prev_t +. alpha *. dt;
          location = (i1, i2);
          precise_point = (x_intersect, 
                          Tensor.get prev_sol i1 |> Tensor.to_float0_exn);
          pre_velocities = (v1, v2);
          post_behavior;
        }
    | [] -> None
    | _ -> failwith "Multiple intersections detected"

  let handle_intersection_region solution x t intersection =
    match intersection with
    | Some isect ->
        let (i1, i2) = isect.location in
        let (x_int, y_int) = isect.precise_point in
        let n = Tensor.size solution 0 in
        let modified = Tensor.copy solution in
        
        (* Apply smoothing around intersection *)
        let smooth_region = 3 in
        for i = max 0 (i1 - smooth_region) to min (n-1) (i2 + smooth_region) do
          let x_i = Tensor.get x i |> Tensor.to_float0_exn in
          let dist = abs_float (x_i -. x_int) in
          let weight = exp (-.(dist ** 2.0) /. 
                          (2.0 *. float_of_int smooth_region ** 2.0)) in
          
          let curr_val = Tensor.get solution i |> Tensor.to_float0_exn in
          let smoothed_val = match isect.post_behavior with
            | `Continuous -> curr_val
            | `Discontinuous slope ->
                let offset = slope *. (x_i -. x_int) in
                curr_val +. weight *. offset
          in
          
          Tensor.fill_float1_idx modified i smoothed_val
        done;
        
        modified
    | None -> solution

  let apply_intersection_barrier_conditions solution x t intersection =
    match intersection with
    | Some isect ->
        let (i1, i2) = isect.location in
        let (x_int, y_int) = isect.precise_point in
        let dx = Tensor.get x 1 |> Tensor.to_float0_exn in
        
        (* Compute one-sided derivatives *)
        let left_deriv = ref 0.0 in
        let right_deriv = ref 0.0 in
        
        if i1 > 0 then begin
          let x_prev = Tensor.get x (i1-1) |> Tensor.to_float0_exn in
          let y_prev = Tensor.get solution (i1-1) |> Tensor.to_float0_exn in
          left_deriv := (y_int -. y_prev) /. (x_int -. x_prev)
        end;
        
        if i2 < Tensor.size x 0 - 1 then begin
          let x_next = Tensor.get x (i2+1) |> Tensor.to_float0_exn in
          let y_next = Tensor.get solution (i2+1) |> Tensor.to_float0_exn in
          right_deriv := (y_next -. y_int) /. (x_next -. x_int)
        end;
        
        Some (!left_deriv, !right_deriv)
    | None -> None
end

module IntersectionTimeStepping = struct
  open Types
  open Torch

  let adapt_timestep dt intersection_info =
    match intersection_info with
    | Some isect ->
        let (v1, v2) = isect.pre_velocities in
        let velocity_diff = abs_float (v2 -. v1) in
        
        (* Reduce timestep based on velocity difference *)
        let reduction_factor = 1.0 /. (1.0 +. velocity_diff) in
        dt *. min 0.5 reduction_factor
    | None -> dt

  let solve_near_intersection a b c v payoff t dt intersection =
    match intersection with
    | Some isect when abs_float (t -. isect.time) < dt ->
        (* Use reduced timestep near intersection *)
        let dt_reduced = dt /. 4.0 in
        let intermediate_steps = 4 in
        let solution = ref (Tensor.copy v) in
        
        for i = 1 to intermediate_steps do
          let curr_t = t +. float_of_int (i-1) *. dt_reduced in
          
          (* Solve with finer resolution *)
          let step_result = DoubleSweepLU.solve a b c !solution payoff in
          match step_result with
          | Ok next_sol ->
              solution := IntersectionPointHandler.handle_intersection_region 
                next_sol v curr_t (Some isect)
          | Error msg -> failwith msg
        done;
        
        Ok !solution
    | _ ->
        (* Normal solution away from intersection *)
        DoubleSweepLU.solve a b c v payoff
end

module IntersectionErrorAnalysis = struct
  open Types
  open Torch

  type error_metrics = {
    max_error: float;
    l2_error: float;
    intersection_error: float option;
    smoothness_violation: float option;
    barrier_preservation: float;
  }

  let compute_intersection_errors solution x t intersection reference =
    match intersection with
    | Some isect ->
        let (i1, i2) = isect.location in
        let n = Tensor.size solution 0 in
        
        (* Local error around intersection *)
        let local_errors = ref [] in
        let smooth_violations = ref [] in
        
        let window = 3 in
        for i = max 0 (i1 - window) to min (n-1) (i2 + window) do
          let sol_i = Tensor.get solution i |> Tensor.to_float0_exn in
          let ref_i = Tensor.get reference i |> Tensor.to_float0_exn in
          local_errors := abs_float (sol_i -. ref_i) :: !local_errors;
          
          if i > 0 then begin
            let sol_prev = Tensor.get solution (i-1) |> Tensor.to_float0_exn in
            let ref_prev = Tensor.get reference (i-1) |> Tensor.to_float0_exn in
            
            let sol_diff = abs_float (sol_i -. sol_prev) in
            let ref_diff = abs_float (ref_i -. ref_prev) in
            smooth_violations := abs_float (sol_diff -. ref_diff) :: !smooth_violations
          end
        done;
        
        (* Global error metrics *)
        let diff = Tensor.sub solution reference in
        let max_error = Tensor.max diff |> Tensor.to_float0_exn in
        let l2_error = Tensor.norm ~p:2 diff |> Tensor.to_float0_exn in
        
        (* Barrier preservation *)
        let barrier_error = ref 0.0 in
        let (x_int, y_int) = isect.precise_point in
        
        (* Interpolate solution at intersection point *)
        let x1 = Tensor.get x i1 |> Tensor.to_float0_exn in
        let x2 = Tensor.get x i2 |> Tensor.to_float0_exn in
        let y1 = Tensor.get solution i1 |> Tensor.to_float0_exn in
        let y2 = Tensor.get solution i2 |> Tensor.to_float0_exn in
        let alpha = (x_int -. x1) /. (x2 -. x1) in
        let sol_at_int = y1 +. alpha *. (y2 -. y1) in
        
        barrier_error := abs_float (sol_at_int -. y_int);
        
        {
          max_error;
          l2_error;
          intersection_error = Some (List.fold_left max 0.0 !local_errors);
          smoothness_violation = 
            if List.length !smooth_violations > 0 
            then Some (List.fold_left max 0.0 !smooth_violations)
            else None;
          barrier_preservation = !barrier_error;
        }
    | None ->
        let diff = Tensor.sub solution reference in
        {
          max_error = Tensor.max diff |> Tensor.to_float0_exn;
          l2_error = Tensor.norm ~p:2 diff |> Tensor.to_float0_exn;
          intersection_error = None;
          smoothness_violation = None;
          barrier_preservation = 0.0;
        }

  let validate_intersection_continuity solution x intersection tol =
    match intersection with
    | Some isect ->
        let (i1, i2) = isect.location in
        let n = Tensor.size solution 0 in
        
        (* Check continuity of solution and derivatives *)
        let continuity_violations = ref [] in
        
        (* Solution continuity *)
        let left_val = Tensor.get solution i1 |> Tensor.to_float0_exn in
        let right_val = Tensor.get solution i2 |> Tensor.to_float0_exn in
        if abs_float (right_val -. left_val) > tol then
          continuity_violations := (`Solution_Jump, abs_float (right_val -. left_val)) 
                                 :: !continuity_violations;
        
        (* First derivative continuity *)
        if i1 > 0 && i2 < n-1 then begin
          let x0 = Tensor.get x (i1-1) |> Tensor.to_float0_exn in
          let x1 = Tensor.get x i1 |> Tensor.to_float0_exn in
          let x2 = Tensor.get x i2 |> Tensor.to_float0_exn in
          let x3 = Tensor.get x (i2+1) |> Tensor.to_float0_exn in
          
          let f0 = Tensor.get solution (i1-1) |> Tensor.to_float0_exn in
          let f1 = left_val in
          let f2 = right_val in
          let f3 = Tensor.get solution (i2+1) |> Tensor.to_float0_exn in
          
          let left_deriv = (f1 -. f0) /. (x1 -. x0) in
          let right_deriv = (f3 -. f2) /. (x3 -. x2) in
          
          if abs_float (right_deriv -. left_deriv) > tol then
            continuity_violations := (`Derivative_Jump, 
                                    abs_float (right_deriv -. left_deriv))
                                   :: !continuity_violations
        end;
        
        !continuity_violations
    | None -> []
end

module BarrierSolver = struct
  open Types
  open Torch

  type barrier_point = {
    location: int;
    value: float;
    time: float;
    derivative: float;
    second_derivative: float;
  }

  type intersection_state = {
    time: float;
    location: float * float;
    pre_intersection_slopes: float * float;
    post_intersection_behavior: [`Continuous | `Discontinuous of float];
    stability_metric: float;
  }

  let compute_barrier_derivatives solution x t =
    let n = Tensor.size solution 0 in
    let dx = Tensor.get x 1 |> Tensor.to_float0_exn in
    
    (* Sixth-order accurate first derivative *)
    let first_derivative i =
      if i < 3 || i > n-4 then
        (* Fourth-order near barriers *)
        let f_m2 = if i >= 2 then Tensor.get solution (i-2) |> Tensor.to_float0_exn else 0.0 in
        let f_m1 = if i >= 1 then Tensor.get solution (i-1) |> Tensor.to_float0_exn else 0.0 in
        let f_p1 = if i < n-1 then Tensor.get solution (i+1) |> Tensor.to_float0_exn else 0.0 in
        let f_p2 = if i < n-2 then Tensor.get solution (i+2) |> Tensor.to_float0_exn else 0.0 in
        (-f_p2 +. 8.*.f_p1 -. 8.*.f_m1 +. f_m2) /. (12.*.dx)
      else
        (* Sixth-order central difference *)
        let f_m3 = Tensor.get solution (i-3) |> Tensor.to_float0_exn in
        let f_m2 = Tensor.get solution (i-2) |> Tensor.to_float0_exn in
        let f_m1 = Tensor.get solution (i-1) |> Tensor.to_float0_exn in
        let f_p1 = Tensor.get solution (i+1) |> Tensor.to_float0_exn in
        let f_p2 = Tensor.get solution (i+2) |> Tensor.to_float0_exn in
        let f_p3 = Tensor.get solution (i+3) |> Tensor.to_float0_exn in
        (f_m3 -. 9.*.f_m2 +. 45.*.f_m1 -. 45.*.f_p1 +. 9.*.f_p2 -. f_p3) /. (60.*.dx)
    in
    
    (* Sixth-order accurate second derivative *)
    let second_derivative i =
      if i < 3 || i > n-4 then
        (* Fourth-order near barriers *)
        let f_m2 = if i >= 2 then Tensor.get solution (i-2) |> Tensor.to_float0_exn else 0.0 in
        let f_m1 = if i >= 1 then Tensor.get solution (i-1) |> Tensor.to_float0_exn else 0.0 in
        let f_i = Tensor.get solution i |> Tensor.to_float0_exn in
        let f_p1 = if i < n-1 then Tensor.get solution (i+1) |> Tensor.to_float0_exn else 0.0 in
        let f_p2 = if i < n-2 then Tensor.get solution (i+2) |> Tensor.to_float0_exn else 0.0 in
        (-f_p2 +. 16.*.f_p1 -. 30.*.f_i +. 16.*.f_m1 -. f_m2) /. (12.*.dx*.dx)
      else
        (* Sixth-order central difference *)
        let f_m3 = Tensor.get solution (i-3) |> Tensor.to_float0_exn in
        let f_m2 = Tensor.get solution (i-2) |> Tensor.to_float0_exn in
        let f_m1 = Tensor.get solution (i-1) |> Tensor.to_float0_exn in
        let f_i = Tensor.get solution i |> Tensor.to_float0_exn in
        let f_p1 = Tensor.get solution (i+1) |> Tensor.to_float0_exn in
        let f_p2 = Tensor.get solution (i+2) |> Tensor.to_float0_exn in
        let f_p3 = Tensor.get solution (i+3) |> Tensor.to_float0_exn in
        (2.*.f_m3 -. 27.*.f_m2 +. 270.*.f_m1 -. 490.*.f_i +. 270.*.f_p1 -. 27.*.f_p2 +. 2.*.f_p3) 
        /. (180.*.dx*.dx)
    in
    
    (first_derivative, second_derivative)

  let locate_barrier solution x payoff tol =
    let n = Tensor.size solution 0 in
    let barriers = ref [] in
    let in_exercise = ref false in
    let start_idx = ref 0 in
    
    for i = 1 to n-2 do
      let val_i = Tensor.get solution i |> Tensor.to_float0_exn in
      let payoff_i = Tensor.get payoff i |> Tensor.to_float0_exn in
      
      if abs_float (val_i -. payoff_i) < tol then begin
        if not !in_exercise then begin
          in_exercise := true;
          start_idx := i;
          
          (* Compute derivatives at barrier *)
          let (first_deriv, second_deriv) = compute_barrier_derivatives solution x 0.0 in
          barriers := {
            location = i;
            value = val_i;
            time = 0.0;
            derivative = first_deriv i;
            second_derivative = second_deriv i;
          } :: !barriers
        end
      end else if !in_exercise then begin
        in_exercise := false;
      end
    done;
    
    List.rev !barriers

  let solve_intersection prev_barriers curr_barriers dt =
    match prev_barriers, curr_barriers with
    | [b1; b2], [b3; b4] when b1.time +. dt = b3.time ->
        let dx = b2.value -. b1.value in
        let dy = b4.value -. b3.value in
        
        if dx *. dy < 0.0 then begin
          let t_intersect = b1.time +. 
            dt *. abs_float(b2.value -. b1.value) /. 
            (abs_float(b4.value -. b3.value) +. abs_float(b2.value -. b1.value)) in
          
          let x_intersect = b1.value +. (b2.value -. b1.value) *. 
            (t_intersect -. b1.time) /. dt in
          
          let slope1 = b2.derivative -. b1.derivative in
          let slope2 = b4.derivative -. b3.derivative in
          
          let post_behavior = 
            if abs_float(b4.second_derivative -. b3.second_derivative) > 1e-6 then
              `Discontinuous (b4.second_derivative -. b3.second_derivative)
            else
              `Continuous in
          
          Some {
            time = t_intersect;
            location = (x_intersect, b1.value +. b1.derivative *. (t_intersect -. b1.time));
            pre_intersection_slopes = (slope1, slope2);
            post_intersection_behavior = post_behavior;
            stability_metric = abs_float(slope1 -. slope2);
          }
        end else None
    | _ -> None

  let solve_barrier_evolution barriers dt =
    let n = List.length barriers in
    if n < 2 then []
    else
      let velocities = ref [] in
      let accelerations = ref [] in
      
      for i = 1 to n-1 do
        let b1 = List.nth barriers (i-1) in
        let b2 = List.nth barriers i in
        
        let v = (b2.value -. b1.value) /. dt in
        velocities := (b1.time, v) :: !velocities;
        
        if i > 1 then begin
          let b0 = List.nth barriers (i-2) in
          let v_prev = (b1.value -. b0.value) /. dt in
          let a = (v -. v_prev) /. dt in
          accelerations := (b1.time, a) :: !accelerations
        end
      done;
      
      List.map2 (fun (t1, v) (t2, a) -> (t1, v, a))
        (List.rev !velocities) (List.rev !accelerations)
end