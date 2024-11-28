open Torch

let integrate ~f ~a ~b ~n =
  let h = (b -. a) /. float_of_int n in
  let sum = ref (f a +. f b) in
  for i = 1 to n - 1 do
    let x = a +. float_of_int i *. h in
    sum := !sum +. (if i mod 2 = 0 then 2.0 else 4.0) *. f x
  done;
  h *. !sum /. 3.0

let derivative ~f ~x ~h =
  (f (x +. h) -. f (x -. h)) /. (2.0 *. h)

let second_derivative ~f ~x ~h =
  (f (x +. h) -. 2.0 *. f x +. f (x -. h)) /. (h *. h)

let solve_tridiagonal a b c r =
  let n = Array.length b in
  let x = Array.make n 0.0 in
  let cp = Array.make n 0.0 in
  let dp = Array.make n 0.0 in
  
  (* Forward sweep *)
  cp.(0) <- c.(0) /. b.(0);
  dp.(0) <- r.(0) /. b.(0);
  
  for i = 1 to n - 1 do
    let m = b.(i) -. a.(i) *. cp.(i-1) in
    cp.(i) <- c.(i) /. m;
    dp.(i) <- (r.(i) -. a.(i) *. dp.(i-1)) /. m
  done;
  
  (* Back substitution *)
  x.(n-1) <- dp.(n-1);
  for i = n - 2 downto 0 do
    x.(i) <- dp.(i) -. cp.(i) *. x.(i+1)
  done;
  x

type t = {
  volatility: float -> float;
  brownian: BrownianMotion.t;
}

let create ~volatility ~brownian = {
  volatility;
  brownian;
}

let simulate slm ~initial_value =
  let bm_path = BrownianMotion.simulate slm.brownian ~initial_value in
  let num_steps = Tensor.size bm_path [0] in
  let path = Tensor.zeros [num_steps] ~device:(Tensor.device bm_path) in
  Tensor.set path [0] (Tensor.of_float initial_value);
  
  for i = 0 to num_steps - 2 do
    let x = Tensor.get path [i] |> Tensor.to_float0_exn in
    let sigma = slm.volatility x in
    let bm_incr = Tensor.get bm_path [i + 1] in
    let next = Tensor.add (Tensor.get path [i]) 
                         (Tensor.mul_scalar bm_incr sigma) in
    Tensor.set path [i + 1] next
  done;
  path

let quadratic_normal ~alpha0 ~alpha1 ~alpha2 x =
  alpha0 +. alpha1 *. x +. alpha2 *. x *. x

let bessel_2d x = exp x

module Scale = struct
  type t = {
    volatility: float -> float;
    phi_up: float -> float;
    phi_down: float -> float;
    lambda: float;
  }

  let create ~volatility ~lambda =
    let compute_phi_up x =
      let rec integrate_up x' acc =
        if x' > 100.0 then acc
        else
          let dx = 0.001 in
          let integrand = 1. /. (volatility x' ** 2.0) in
          integrate_up (x' +. dx) (acc +. integrand *. dx)
      in
      exp (lambda *. integrate_up x 0.0)
    in

    let compute_phi_down x =
      let rec integrate_down x' acc =
        if x' < -100.0 then acc
        else
          let dx = -0.001 in
          let integrand = 1. /. (volatility x' ** 2.0) in
          integrate_down (x' +. dx) (acc +. integrand *. dx)
      in
      exp (lambda *. integrate_down x 0.0)
    in

    {
      volatility;
      phi_up = compute_phi_up;
      phi_down = compute_phi_down;
      lambda;
    }

  let phi_combined scale x =
    scale.phi_up x +. scale.phi_down x
end

module BrownianMotion = struct
  type t = {
    dt: float;
    num_steps: int;
    device: Device.t;
  }

  let create ~dt ~num_steps ~device = {
    dt;
    num_steps;
    device;
  }

  let simulate bm ~initial_value =
    let sqrt_dt = sqrt bm.dt in
    let increments = Tensor.randn [bm.num_steps] ~device:bm.device in
    let scaled_increments = Tensor.mul_scalar increments sqrt_dt in
    let path = Tensor.zeros [bm.num_steps + 1] ~device:bm.device in
    Tensor.set path [0] (Tensor.of_float initial_value);
    
    for i = 0 to bm.num_steps - 1 do
      let prev = Tensor.get path [i] in
      let incr = Tensor.get scaled_increments [i] in
      let next = Tensor.add prev incr in
      Tensor.set path [i + 1] next
    done;
    path
end

module HarmonicFunction = struct
  type t = {
    lambda: float;
    scale: float -> float;
    volatility: float -> float;
  }

  let create ~lambda ~scale ~volatility = {
    lambda;
    scale;
    volatility;
  }

  let evaluate h x =
    let sigma = h.volatility x in
    let s = h.scale x in
    (h.lambda *. s) /. (sigma *. sigma)

  let is_strictly_convex h x1 x2 a =
    let u1 = evaluate h x1 in
    let u2 = evaluate h x2 in
    let xa = a *. x1 +. (1. -. a) *. x2 in
    let ua = evaluate h xa in
    ua < a *. u1 +. (1. -. a) *. u2
end

module PDE = struct
  type solution = {
    value: float -> float -> float;
    gradient: float -> float -> float;
  }

  let solve_weak_pde ~volatility ~initial_condition ~lambda ~t_max ~x_grid =
    let dx = x_grid.(1) -. x_grid.(0) in
    let dt = t_max /. float_of_int (Array.length x_grid) in
    
    let solution = Array.make_matrix (Array.length x_grid) 2 0.0 in
    Array.iteri (fun i x -> solution.(i).(0) <- initial_condition x) x_grid;
    
    for t = 1 to Array.length x_grid - 1 do
      for i = 1 to Array.length x_grid - 2 do
        let x = x_grid.(i) in
        let sigma = volatility x in
        let sigma_sq = sigma *. sigma in
        
        let d2u_dx2 = (solution.(i+1).(t-1) -. 2.0 *. solution.(i).(t-1) 
                       +. solution.(i-1).(t-1)) /. (dx *. dx) in
        
        solution.(i).(t) <- solution.(i).(t-1) +. 
                           0.5 *. sigma_sq *. d2u_dx2 *. dt
      done
    done;
    
    let value t x =
      let t_idx = int_of_float (t /. dt) in
      let x_idx = int_of_float ((x -. x_grid.(0)) /. dx) in
      solution.(x_idx).(t_idx)
    in
    
    let gradient t x =
      let t_idx = int_of_float (t /. dt) in
      let x_idx = int_of_float ((x -. x_grid.(0)) /. dx) in
      (solution.(x_idx+1).(t_idx) -. solution.(x_idx-1).(t_idx)) /. (2.0 *. dx)
    in
    
    { value; gradient }
end

module HigherMoments = struct
  type moment_bound = {
    p: float;
    c: float;
  }

  let check_polynomial_growth ~bound ~f x =
    abs_float (f x) <= bound.c *. (1. +. abs_float x ** bound.p)

  module RegularizedMartingale = struct
    type t = {
      original_process: StrictLocalMartingale.t;
      h_star: float -> float -> float;
      terminal_time: float;
    }

    let create ~process ~T ~h_star = {
      original_process = process;
      h_star;
      terminal_time = T;
    }

    let dynamics mart t x =
      let dx = mart.h_star (mart.terminal_time -. t) x in
      let sigma = mart.original_process.volatility x in
      dx *. sigma

    let simulate mart ~initial_value ~num_steps =
      let dt = mart.terminal_time /. float_of_int num_steps in
      let device = Device.Cpu in
      let path = Tensor.zeros [num_steps + 1] ~device in
      Tensor.set path [0] (Tensor.of_float initial_value);
      
      let bm = BrownianMotion.create ~dt ~num_steps ~device in
      let noise = BrownianMotion.simulate bm ~initial_value:0.0 in
      
      for i = 0 to num_steps - 1 do
        let t = float_of_int i *. dt in
        let curr_x = Tensor.get path [i] |> Tensor.to_float0_exn in
        let drift = dynamics mart t curr_x in
        let dw = Tensor.get noise [i+1] |> Tensor.to_float0_exn in
        let next_x = curr_x +. drift *. sqrt dt *. dw in
        Tensor.set path [i + 1] (Tensor.of_float next_x)
      done;
      path
  end
end

module InverseBessel3D = struct
  type t = {
    y: float;
  }

  let create ~initial_value = {
    y = initial_value;
  }

  let volatility proc x =
    1. /. (x *. x)

  let simulate proc ~dt ~num_steps =
    let device = Device.Cpu in
    let path = Tensor.zeros [num_steps + 1] ~device in
    Tensor.set path [0] (Tensor.of_float proc.y);
    
    let bm = BrownianMotion.create ~dt ~num_steps ~device in
    let noise = BrownianMotion.simulate bm ~initial_value:0.0 in
    
    for i = 0 to num_steps - 1 do
      let curr_y = Tensor.get path [i] |> Tensor.to_float0_exn in
      let sigma = volatility proc curr_y in
      let dw = Tensor.get noise [i+1] |> Tensor.to_float0_exn in
      let next_y = curr_y -. sigma *. curr_y *. curr_y *. sqrt dt *. dw in
      Tensor.set path [i + 1] (Tensor.of_float next_y)
    done;
    path

  let second_moment proc t =
    let y0 = proc.y in
    y0 *. y0 +. t

  let quadratic_variation proc t =
    float_of_int max_int  (* Infinite quadratic variation *)
end

module MartingaleAnalysis = struct
  type martingale_type = 
    | TrueMartingale
    | StrictLocalMartingale
    | Submartingale
    | Unknown

  let classify_process ~volatility ~initial_value ~t_max =
    let check_kotani_condition () =
      let integrand x = x /. (volatility x ** 2.0) in
      let upper_integral = Utilities.integrate ~f:integrand ~a:0.0 ~b:100.0 ~n:1000 in
      let lower_integral = Utilities.integrate ~f:integrand ~a:(-100.0) ~b:0.0 ~n:1000 in
      (upper_integral < infinity, lower_integral < infinity)
    in
    
    match check_kotani_condition () with
    | (true, false) | (false, true) -> StrictLocalMartingale
    | (true, true) -> TrueMartingale
    | (false, false) -> Unknown
end

module NumericalAnalysis = struct
  module AdaptiveSteps = struct
    type error_control = {
      abs_tol: float;
      rel_tol: float;
      safety_factor: float;
    }

    let compute_optimal_step ~current_step ~error ~control ~order =
      let scale = min 
        (control.safety_factor *. (control.abs_tol /. error) ** (1.0 /. float_of_int order))
        2.0 
      in
      current_step *. scale

    let runge_kutta_adaptive ~system ~initial ~t_span ~control =
      let rec step times values current_step =
        if List.length times >= 1000 || List.hd times >= snd t_span then
          (List.rev times, List.rev values)
        else
          let t = List.hd times in
          let y = List.hd values in
          
          (* RK4 step *)
          let k1 = system t y in
          let k2 = system (t +. 0.5 *. current_step) 
                        (y +. 0.5 *. current_step *. k1) in
          let k3 = system (t +. 0.5 *. current_step)
                        (y +. 0.5 *. current_step *. k2) in
          let k4 = system (t +. current_step)
                        (y +. current_step *. k3) in
          
          let dy = (k1 +. 2.0 *. k2 +. 2.0 *. k3 +. k4) /. 6.0 in
          let error = abs_float (dy *. current_step) in
          
          let new_step = compute_optimal_step 
            ~current_step ~error ~control ~order:4 in
          
          if error <= control.abs_tol then
            let new_t = t +. current_step in
            let new_y = y +. current_step *. dy in
            step (new_t :: times) (new_y :: values) new_step
          else
            step times values new_step
      in
      step [fst t_span] [initial] ((snd t_span -. fst t_span) /. 100.0)
  end

  module SpectralMethods = struct
    let chebyshev_diff_matrix n =
      let matrix = Array.make_matrix n n 0.0 in
      let nodes = Array.init n (fun i ->
        cos (Float.pi *. float_of_int i /. float_of_int (n - 1))
      ) in
      
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          if i <> j then
            let ci = if i = 0 || i = n - 1 then 2.0 else 1.0 in
            let cj = if j = 0 || j = n - 1 then 2.0 else 1.0 in
            matrix.(i).(j) <- ci /. cj *. 
              ((-1.0) ** float_of_int (i + j)) /.
              (nodes.(i) -. nodes.(j))
        done
      done;
      
      for i = 0 to n - 1 do
        let sum = ref 0.0 in
        for j = 0 to n - 1 do
          if j <> i then sum := !sum +. matrix.(i).(j)
        done;
        matrix.(i).(i) <- -.!sum
      done;
      matrix

    let solve_spectral_pde ~pde_func ~initial_condition ~boundary_conditions ~grid =
      let n = grid.PDE.value 0.0 1.0 |> int_of_float in
      let diff_matrix = chebyshev_diff_matrix n in
      let solution = Array.make_matrix n n 0.0 in
      
      (* Initialize with initial condition *)
      for i = 0 to n - 1 do
        solution.(i).(0) <- initial_condition 
          (grid.PDE.value (float_of_int i) 0.0)
      done;
      
      (* Time stepping *)
      let dt = 1.0 /. float_of_int (n - 1) in
      for j = 0 to n - 2 do
        (* Apply spectral differentiation *)
        let diff = Array.init n (fun i ->
          let sum = ref 0.0 in
          for k = 0 to n - 1 do
            sum := !sum +. diff_matrix.(i).(k) *. solution.(k).(j)
          done;
          !sum
        ) in
        
        (* Update interior points *)
        for i = 1 to n - 2 do
          let t = float_of_int j *. dt in
          let x = grid.PDE.value (float_of_int i) 0.0 in
          solution.(i).(j + 1) <- solution.(i).(j) +. 
            dt *. pde_func t x solution.(i).(j) diff.(i)
        done;
        
        (* Apply boundary conditions *)
        solution.(0).(j + 1) <- (fst boundary_conditions) 
          (float_of_int (j + 1) *. dt);
        solution.(n - 1).(j + 1) <- (snd boundary_conditions)
          (float_of_int (j + 1) *. dt)
      done;
      solution
  end
end