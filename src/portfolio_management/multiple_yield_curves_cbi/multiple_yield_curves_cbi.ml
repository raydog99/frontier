open Torch

(* Filtration representation *)
module Filtration = struct
  type t = {
    time_points: float array;
    sub_sigma_fields: (float array) array;
  }

  let create time_points =
    let n = Array.length time_points in
    let sub_sigma_fields = Array.make n [||] in
    for i = 0 to n - 1 do
      sub_sigma_fields.(i) <- Array.sub time_points 0 (i + 1)
    done;
    {time_points; sub_sigma_fields}

  let is_adapted filtration process =
    let n = Array.length filtration.time_points in
    let is_measurable t values =
      let idx = ref 0 in
      while !idx < n && filtration.time_points.(!idx) <= t do
        incr idx
      done;
      Array.length values <= !idx
    in
    Array.for_all2 is_measurable filtration.time_points process
end

(* Sigma-finite measure *)
module Measure = struct
  type t = {
    domain: float * float;
    density: float -> float;
    total_mass: float option;
  }

  let lebesgue (a, b) = {
    domain = (a, b);
    density = (fun _ -> 1.0);
    total_mass = Some (b -. a);
  }

  let counting points = {
    domain = (min_float, max_float);
    density = (fun x -> if Array.mem x points then 1.0 else 0.0);
    total_mass = Some (float_of_int (Array.length points));
  }

  let integrate measure f =
    let (a, b) = measure.domain in
    let n = 1000 in 
    let dx = (b -. a) /. float_of_int n in
    let sum = ref 0.0 in
    for i = 0 to n - 1 do
      let x = a +. dx *. float_of_int i in
      sum := !sum +. f x *. measure.density x *. dx
    done;
    !sum

  let is_finite measure =
    match measure.total_mass with
    | Some mass -> mass < infinity
    | None -> false
end

(* Random measure *)
module RandomMeasure = struct
  type t = {
    intensity: Measure.t;
    generator: unit -> float;
  }

  let poisson intensity =
    let generator () =
      let u = Random.float 1.0 in
      -. log u /. Measure.integrate intensity (fun _ -> 1.0)
    in
    {intensity; generator}

  let simulate measure t n =
    let jumps = ref [] in
    let current_time = ref 0.0 in
    while !current_time < t do
      let next_jump = measure.generator () in
      current_time := !current_time +. next_jump;
      if !current_time < t then
        jumps := !current_time :: !jumps
    done;
    Array.of_list (List.rev !jumps)

  let compensate measure =
    let compensated_intensity = {
      measure.intensity with
      density = (fun x -> 
        measure.intensity.density x -. 
        Measure.integrate measure.intensity (fun _ -> 1.0));
    } in
    {measure with intensity = compensated_intensity}
end

(* Stochastic integral *)
module StochasticIntegral = struct
  let brownian_motion t n dt =
    let steps = int_of_float (t /. dt) in
    let path = Array.make (steps + 1) 0.0 in
    for i = 1 to steps do
      let dW = Tensor.randn [|1|] |> Tensor.get [|0|] in
      path.(i) <- path.(i-1) +. sqrt dt *. dW
    done;
    path

  let ito_integral process brownian t dt =
    let steps = int_of_float (t /. dt) in
    let integral = ref 0.0 in
    for i = 0 to steps - 1 do
      let dW = brownian.(i+1) -. brownian.(i) in
      integral := !integral +. process.(i) *. dW
    done;
    !integral

  let poisson_integral process measure t =
    let jumps = RandomMeasure.simulate measure t 1000 in
    Array.fold_left (fun acc jump_time ->
      let idx = int_of_float (jump_time /. 0.001) in
      acc +. process.(idx)
    ) 0.0 jumps
end

(* Probability space *)
module Space = struct
  type t = {
    sample_space: Measure.t;
    filtration: Filtration.t;
    probability: float -> float;
  }

  let create sample_space filtration probability =
    let total_prob = Measure.integrate sample_space probability in
    {sample_space; filtration; probability}

  let expectation space f =
    Measure.integrate space.sample_space 
      (fun x -> f x *. space.probability x)

  let conditional_expectation space f condition =
    let conditional_prob = 
      space.probability condition |> max 1e-10 in
    let integrand x =
      if condition x then f x *. space.probability x
      else 0.0
    in
    Measure.integrate space.sample_space integrand /. conditional_prob
end

(* Stochastic process *)
module Process = struct
  type 'a t = {
    paths: ('a array) array;
    time_points: float array;
    filtration: Filtration.t;
  }

  let create paths time_points =
    let filtration = Filtration.create time_points in
    if not (Filtration.is_adapted filtration 
            (Array.map (fun p -> p.(0)) paths)) then
      raise (Invalid_argument "Process must be adapted");
    {paths; time_points; filtration}

  let is_martingale process prob_space =
    let n = Array.length process.time_points in
    let check_martingale_property t =
      if t >= n - 1 then true
      else
        let current_values = 
          Array.map (fun p -> p.(t)) process.paths in
        let next_values = 
          Array.map (fun p -> p.(t+1)) process.paths in
        let conditional_exp = Space.conditional_expectation
          prob_space
          (fun x -> next_values.(int_of_float x))
          (fun x -> x < float_of_int (Array.length current_values))
        in
        abs_float (conditional_exp -. float_of_int t) < 1e-10
    in
    Array.init (n-1) (fun i -> i)
    |> Array.for_all check_martingale_property

  let quadratic_variation process =
    let n = Array.length process.time_points in
    let qv = Array.make n 0.0 in
    for i = 1 to n - 1 do
      let increments = Array.map (fun p ->
        p.(i) -. p.(i-1)
      ) process.paths in
      qv.(i) <- qv.(i-1) +. 
        Array.fold_left (fun acc x -> acc +. x *. x) 
          0.0 increments
    done;
    qv

  let is_continuous process epsilon =
    let check_path path =
      let n = Array.length path in
      let max_jump = ref 0.0 in
      for i = 1 to n - 1 do
        max_jump := max !max_jump (abs_float (path.(i) -. path.(i-1)))
      done;
      !max_jump < epsilon
    in
    Array.for_all check_path process.paths
end

(* Lévy process *)
module LevyProcess = struct
  type t = float Process.t

  let create increments time_points =
    let n = Array.length time_points in
    let m = Array.length increments in
    let paths = Array.make m [||] in
    for i = 0 to m - 1 do
      paths.(i) <- Array.make n 0.0;
      for j = 1 to n - 1 do
        paths.(i).(j) <- paths.(i).(j-1) +. increments.(i)
      done
    done;
    Process.create paths time_points
end

(* Branching mechanism *)
module Branching = struct
  type parameters = {
    b: float;           (* Drift coefficient *)
    sigma: float;       (* Diffusion coefficient *)
    pi: Measure.t;      (* Jump measure *)
  }

  (* Compute branching mechanism φ *)
  let phi params z =
    let linear = params.b *. z in
    let diffusion = params.sigma ** 2.0 *. z ** 2.0 /. 2.0 in
    let jump_integral = 
      Measure.integrate params.pi (fun u ->
        exp (z *. u) -. 1.0 -. z *. u
      )
    in
    linear +. diffusion +. jump_integral

  (* Check subcritical condition *)
  let is_subcritical params =
    let deriv_at_zero = params.b +. 
      Measure.integrate params.pi (fun u -> u) in
    deriv_at_zero > 0.0

  (* Compute domain Y *)
  let compute_domain params =
    let rec binary_search a b epsilon =
      if b -. a < epsilon then a
      else
        let mid = (a +. b) /. 2.0 in
        try
          let _ = phi params mid in
          binary_search a mid epsilon
        with _ ->
          binary_search mid b epsilon
    in
    binary_search (-1000.0) 1000.0 1e-6
end

(* Immigration rate *)
module Immigration = struct
  type parameters = {
    beta: float;        (* Linear coefficient *)
    nu: Measure.t;      (* Jump measure *)
  }

  (* Compute immigration rate ψ *)
  let psi params z =
    let linear = params.beta *. z in
    let jump_integral =
      Measure.integrate params.nu (fun u ->
        1.0 -. exp (-. z *. u)
      )
    in
    linear +. jump_integral

  let has_finite_mean params =
    Measure.integrate params.nu (fun u -> u) < infinity
end

(* CBI process *)
module Process = struct
  type parameters = {
    branching: Branching.parameters;
    immigration: Immigration.parameters;
    x0: float;         (* Initial value *)
  }

  (* ODE solver *)
  module ODESolver = struct
    type solution = {
      times: float array;
      values: float array;
    }

    (* Solve ODE using 4th order Runge-Kutta *)
    let solve params p q t_end =
      let dt = 0.001 in
      let n = int_of_float (t_end /. dt) in
      
      let times = Array.init n (fun i -> 
        float_of_int i *. dt) in
      let values = Array.make n p in
      
      for i = 0 to n - 2 do
        let t = times.(i) in
        let v = values.(i) in
        
        let k1 = q -. Branching.phi params.branching v in
        let k2 = q -. Branching.phi params.branching 
                  (v +. dt *. k1 /. 2.0) in
        let k3 = q -. Branching.phi params.branching 
                  (v +. dt *. k2 /. 2.0) in
        let k4 = q -. Branching.phi params.branching 
                  (v +. dt *. k3) in
        
        values.(i + 1) <- v +. dt *. 
          (k1 +. 2.0 *. k2 +. 2.0 *. k3 +. k4) /. 6.0
      done;
      
      {times; values}
  end

  (* Compute transition probability *)
  let transition_probability params t x y =
    let v_solution = ODESolver.solve params y 0.0 t in
    let last_v = v_solution.values.(Array.length v_solution.values - 1) in
    
    let psi_integral =
      Array.fold_left2 (fun acc t v ->
        acc +. Immigration.psi params.immigration v *. 0.001
      ) 0.0 v_solution.times v_solution.values
    in
    
    exp (x *. last_v -. psi_integral)

  (* Compute Laplace transform *)
  let laplace_transform params t p =
    let v_solution = ODESolver.solve params p 0.0 t in
    let last_v = v_solution.values.(Array.length v_solution.values - 1) in
    
    let psi_integral =
      Array.fold_left2 (fun acc t v ->
        acc +. Immigration.psi params.immigration v *. 0.001
      ) 0.0 v_solution.times v_solution.values
    in
    
    exp (-. params.x0 *. last_v -. psi_integral)

  (* Compute characteristic function *)
  let characteristic_function params t u =
    let open Complex in
    let p = {re = 0.0; im = u} in
    
    let v_solution = ODESolver.solve params p 0.0 t in
    let last_v = v_solution.values.(Array.length v_solution.values - 1) in
    
    let psi_integral =
      Array.fold_left2 (fun acc t v ->
        acc +. Immigration.psi params.immigration v *. 0.001
      ) 0.0 v_solution.times v_solution.values
    in
    
    exp (mul (float_to_scalar (-. params.x0)) last_v -. psi_integral)

  (* Compute lifetime T^(p,q) *)
  let compute_lifetime params p q =
    let p_q = ref infinity in
    
    let rec binary_search a b epsilon =
      if b -. a < epsilon then a
      else
        let mid = (a +. b) /. 2.0 in
        let phi_mid = Branching.phi params.branching mid in
        if q -. phi_mid >= 0.0 then
          binary_search a mid epsilon
        else
          binary_search mid b epsilon
    in
    
    let domain = Branching.compute_domain params.branching in
    p_q := binary_search domain infinity 1e-6;
    
    if p <= !p_q then infinity
    else
      (* Compute integral *)
      let integrand y =
        1.0 /. (Branching.phi params.branching y -. q)
      in
      
      let rec integrate a b n acc =
        if n = 0 then acc
        else
          let dy = (b -. a) /. float_of_int n in
          let y = a +. dy *. float_of_int (n - 1) in
          integrate a b (n - 1) (acc +. integrand y *. dy)
      in
      
      integrate domain p 1000 0.0

  (* Simulate CBI process paths *)
  let simulate params t dt n =
    let steps = int_of_float (t /. dt) in
    let paths = ref [] in
    let current = ref (Tensor.full [|n|] params.x0) in
    
    for _ = 1 to steps do
      let dW = Tensor.randn [|n|] in
      let dM = RandomMeasure.simulate 
        (RandomMeasure.poisson params.branching.pi) dt n in
      
      let drift = Tensor.(mul (float_to_scalar params.branching.b) !current) in
      let diffusion = Tensor.(mul (sqrt !current) dW) in
      let jumps = Tensor.(mul !current (of_float_array dM)) in
      
      current := Tensor.(max (add !current 
        (add (mul drift (float_to_scalar dt))
        (add (mul diffusion (float_to_scalar (sqrt dt))) jumps)))
        (zeros_like !current));
      
      paths := !current :: !paths
    done;
    
    List.rev !paths
end

(* Multi-curve model *)
module MultiCurveModel = struct
  (* Model parameters *)
  type model_parameters = {
    ell: float -> float;          (* Time-dependent function *)
    lambda: Tensor.t;             (* Vector in R^d_+ *)
    c: int -> float -> float;     (* Family of functions *)
    gamma: int -> Tensor.t;       (* Family of vectors *)
    cbi_params: Process.parameters; (* CBI process parameters *)
  }

  (* Term structure *)
  module TermStructure = struct
    type forward_curve = {
      tenors: float array;
      rates: float array;
      interpolator: float -> float;
    }

    let create_forward_curve tenors rates =
      let interpolator t =
        let rec find_interval i =
          if i >= Array.length tenors - 1 then i - 1
          else if t <= tenors.(i+1) then i
          else find_interval (i + 1)
        in
        let i = find_interval 0 in
        let t0, t1 = tenors.(i), tenors.(i+1) in
        let r0, r1 = rates.(i), rates.(i+1) in
        r0 +. (r1 -. r0) *. (t -. t0) /. (t1 -. t0)
      in
      {tenors; rates; interpolator}

    let discount_factor curve t T =
      let rec integrate acc t' dt =
        if t' >= T then acc
        else
          let r = curve.interpolator t' in
          integrate (acc -. r *. dt) (t' +. dt) dt
      in
      exp (integrate 0.0 t 0.001)
  end

  (* Forward rate dynamics *)
  module ForwardRates = struct
    (* Forward IBOR rate *)
    let forward_rate model_params t T delta x =
      let expectation = Process.characteristic_function 
        model_params.cbi_params (T -. t) 1.0 in
      Complex.(expectation.re)

    (* Short rate *)
    let short_rate model_params t x =
      let lambda_term = Tensor.(dot model_params.lambda x) in
      model_params.ell t +. Tensor.get lambda_term [|0|]

    (* Log spread *)
    let log_spread model_params i t x =
      let gamma_term = Tensor.(dot (model_params.gamma i) x) in
      model_params.c i t +. Tensor.get gamma_term [|0|]
  end

  (* Spread dynamics *)
  module Spreads = struct
    (* Spot multiplicative spread *)
    let spot_spread model_params t delta x =
      let l_ibor = ForwardRates.forward_rate model_params t t delta x in
      let l_ois = ForwardRates.forward_rate model_params t t delta x in
      (1.0 +. delta *. l_ibor) /. (1.0 +. delta *. l_ois)

    (* Forward multiplicative spread *)
    let forward_spread model_params t T delta x =
      let l_ibor_fwd = ForwardRates.forward_rate 
        model_params t T delta x in
      let l_ois_fwd = ForwardRates.forward_rate 
        model_params t T delta x in
      (1.0 +. delta *. l_ibor_fwd) /. (1.0 +. delta *. l_ois_fwd)

    (* Verify spread monotonicity *)
    let check_monotonicity model_params =
      let check_tenor i =
        let gamma_i = model_params.gamma i in
        let n = Tensor.size gamma_i [|0|] in
        let is_positive = ref true in
        for j = 0 to n - 1 do
          is_positive := !is_positive && 
            (Tensor.get gamma_i [|j|] >= 0.0)
        done;
        !is_positive && model_params.c i 0.0 >= 0.0
      in
      let m = 10 (* Number of tenors to check *) in
      let monotonic = ref true in
      for i = 0 to m - 2 do
        monotonic := !monotonic && check_tenor i && 
          (model_params.c i 0.0 <= model_params.c (i+1) 0.0)
      done;
      !monotonic
  end
end

(* Pricing module *)
module Pricing = struct
  (* Pricing result type *)
  type pricing_result = {
    price: float;
    delta: float option;
    gamma: float option;
    vega: float option;
    error_bound: float option;
  }

  (* FFT-based pricing *)
  module FFT = struct
    (* Modified characteristic function *)
    let modified_char_fn model_params t T delta zeta x =
      let b_T = TermStructure.discount_factor 
        {tenors=[|t; T|]; rates=[|0.0; 0.0|]; 
         interpolator=(fun _ -> 0.0)} t T in
      
      let phi = Process.characteristic_function 
        model_params.cbi_params (T -. t) 
        Complex.{re=0.0; im=zeta} in
      
      Complex.{
        re = b_T *. phi.re;
        im = b_T *. phi.im;
      }

    (* Grid parameters *)
    type grid_params = {
      n_points: int;
      eta: float;      (* Integration spacing *)
      alpha: float;    (* Dampening parameter *)
    }

    (* Create FFT grid *)
    let create_grid params =
      let v = Array.init params.n_points (fun j ->
        float_of_int j *. params.eta
      ) in
      let k = Array.init params.n_points (fun j ->
        -. Float.pi /. params.eta +. 
        2.0 *. Float.pi *. float_of_int j /. 
        (params.eta *. float_of_int params.n_points)
      ) in
      (v, k)

    (* Compute FFT inputs *)
    let compute_fft_inputs model_params t T delta grid_params =
      let v, k = create_grid grid_params in
      
      Array.mapi (fun j v_j ->
        let cf = modified_char_fn model_params t T delta v_j 
          (Tensor.zeros [|1|]) in
        
        let damp = exp (-. grid_params.alpha *. v_j) in
        
        Complex.{
          re = damp *. cf.re;
          im = damp *. cf.im;
        }
      ) v

    (* FFT implementation *)
    let fft inputs =
      let n = Array.length inputs in
      let outputs = Array.make n Complex.{re=0.0; im=0.0} in
      
      for k = 0 to n - 1 do
        for j = 0 to n - 1 do
          let angle = -2.0 *. Float.pi *. float_of_int (j * k) /. 
                     float_of_int n in
          let term = Complex.{
            re = cos angle;
            im = sin angle;
          } in
          let prod = Complex.mul inputs.(j) term in
          outputs.(k) <- Complex.{
            re = outputs.(k).re +. prod.re;
            im = outputs.(k).im +. prod.im;
          }
        done
      done;
      
      outputs

    (* Price caplet using FFT *)
    let price_caplet model_params t T delta K =
      let grid_params = {
        n_points = 4096;
        eta = 0.25;
        alpha = 1.5;
      } in
      
      let inputs = compute_fft_inputs model_params t T delta 
        grid_params in
      
      let outputs = fft inputs in
      
      let _, k = create_grid grid_params in
      let log_K = log K in
      
      let rec find_index i =
        if i >= Array.length k - 1 then i - 1
        else if log_K <= k.(i+1) then i
        else find_index (i + 1)
      in
      
      let i = find_index 0 in
      let k0, k1 = k.(i), k.(i+1) in
      let p0, p1 = outputs.(i).re, outputs.(i+1).re in
      
      let alpha = (log_K -. k0) /. (k1 -. k0) in
      p0 *. (1.0 -. alpha) +. p1 *. alpha
  end

  (* Quantization-based pricing *)
  module Quantization = struct
    (* Optimal quantization grid *)
    type grid = {
      points: float array;
      weights: float array;
    }

    (* L^p distance *)
    let lp_distance grid target_dist p =
      Array.fold_left2 (fun acc point weight ->
        acc +. weight *. (abs_float (point -. target_dist)) ** p
      ) 0.0 grid.points grid.weights ** (1.0 /. p)

    (* Optimal quantization *)
    let optimize_grid initial_grid target_dist p max_iter =
      let rec iterate grid iter =
        if iter >= max_iter then grid
        else
          (* Compute gradients *)
          let gradients = Array.map (fun point ->
            let diff = point -. target_dist in
            p *. sign_float diff *. abs_float diff ** (p -. 1.0)
          ) grid.points in
          
          (* Update points *)
          let new_points = Array.map2 (fun point grad ->
            point -. 0.01 *. grad
          ) grid.points gradients in
          
          (* Update weights using Lloyd's algorithm *)
          let new_weights = Array.map (fun point ->
            let cell_prob = exp (-. (point -. target_dist) ** 2.0 /. 2.0) in
            cell_prob /. (Array.fold_left (+.) 0.0 
              (Array.map (fun p -> 
                exp (-. (p -. target_dist) ** 2.0 /. 2.0)
              ) new_points))
          ) new_points in
          
          iterate {points = new_points; weights = new_weights} (iter + 1)
      in
      iterate initial_grid 0

    (* Companion weights *)
    let companion_weights grid model_params T =
      Array.mapi (fun j point ->
        let prob_leq = Array.fold_lefti (fun acc i p ->
          if i <= j then
            acc +. Process.transition_probability 
              model_params.cbi_params T point p
          else acc
        ) 0.0 grid.points in
        prob_leq *. grid.weights.(j)
      ) grid.points

    (* Price caplet using quantization *)
    let price_caplet model_params t T delta K =
      (* Create initial grid *)
      let initial_grid = {
        points = Array.init 32 (fun i -> 
          -5.0 +. 10.0 *. float_of_int i /. 31.0);
        weights = Array.make 32 (1.0 /. 32.0);
      } in
      
      (* Target distribution from terminal value *)
      let target_dist = Process.laplace_transform 
        model_params.cbi_params T 1.0 in
      
      (* Optimize grid *)
      let grid = optimize_grid initial_grid target_dist 2.0 100 in
      
      (* Compute weights *)
      let weights = companion_weights grid model_params T in
      
      (* Compute price *)
      Array.fold_left2 (fun acc point weight ->
        let payoff = max (exp point -. K) 0.0 in
        acc +. payoff *. weight
      ) 0.0 grid.points weights
  end
end

(* Calibration framework *)
module Calibration = struct
  (* Market data types *)
  type market_caplet = {
    tenor: float;
    maturity: float;
    strike: float;
    price: float;
    bid: float option;
    ask: float option;
  }

  type market_data = {
    observation_date: float;
    ois_curve: TermStructure.forward_curve;
    caplets: market_caplet array;
    tenors: float array;
  }

  (* Divergence function *)
  module Divergence = struct
    type weights = {
      price_weight: float;
      spread_weight: float;
      regularization: float;
    }

    (* Price-based divergence *)
    let price_divergence model_params market_data =
      Array.fold_left (fun acc caplet ->
        let model_price = Pricing.FFT.price_caplet model_params 
          market_data.observation_date
          caplet.maturity caplet.tenor caplet.strike in
        acc +. (model_price -. caplet.price) ** 2.0
      ) 0.0 market_data.caplets

    (* Spread-based divergence *)
    let spread_divergence model_params market_data =
      Array.fold_left (fun acc tenor ->
        let model_spread = Spreads.spot_spread model_params
          market_data.observation_date tenor 
          (Tensor.zeros [|1|]) in
        let market_spread = ForwardRates.forward_rate model_params
          market_data.observation_date
          market_data.observation_date tenor 
          (Tensor.zeros [|1|]) in
        acc +. (model_spread -. market_spread) ** 2.0
      ) 0.0 market_data.tenors

    (* Regularization term *)
    let regularization model_params =
      let params = model_params.cbi_params in
      params.Process.branching.sigma ** 2.0 +.
      params.immigration.beta ** 2.0

    (* Total divergence function *)
    let total_divergence model_params market_data weights =
      weights.price_weight *. price_divergence model_params market_data +.
      weights.spread_weight *. spread_divergence model_params market_data +.
      weights.regularization *. regularization model_params
  end

  (* Parameter constraints *)
  module Constraints = struct
    type bounds = {
      lower: float;
      upper: float;
    }

    type parameter_constraints = {
      alpha: bounds;
      theta: bounds;
      b: bounds;
      sigma: bounds;
      eta: bounds;
    }

    let default_constraints = {
      alpha = {lower = 1.1; upper = 1.9};
      theta = {lower = 0.1; upper = 10.0};
      b = {lower = 0.0; upper = 10.0};
      sigma = {lower = 0.0; upper = 2.0};
      eta = {lower = 0.0; upper = 5.0}
    }

    let apply_constraints params constraints =
      let clip value bounds =
        max bounds.lower (min value bounds.upper)
      in
      {params with
        Process.branching = {
          params.Process.branching with
          b = clip params.Process.branching.b constraints.b;
          sigma = clip params.Process.branching.sigma 
                  constraints.sigma;
        };
        immigration = {
          params.Process.immigration with
          beta = clip params.Process.immigration.beta 
                 constraints.eta;
        }
      }
  end

  (* Optimization *)
  module Optimization = struct
    type config = {
      learning_rate: float;
      max_iterations: int;
      tolerance: float;
      momentum: float;
    }

    (* Gradient descent with momentum *)
    let optimize initial_params market_data weights constraints config =
      let rec optimize_step params velocity iter best_divergence =
        if iter >= config.max_iterations then params
        else
          (* Compute divergence and gradients *)
          let divergence = Divergence.total_divergence params market_data weights in
          
          if divergence < best_divergence -. config.tolerance then
            (* Compute gradients *)
            let grads = grad_of_fn (fun p -> 
              Divergence.total_divergence p market_data weights) params in
            
            (* Update velocity *)
            let new_velocity = 
              add (mul velocity (float_to_scalar config.momentum))
                  (mul grads (float_to_scalar config.learning_rate)) in
            
            (* Update parameters *)
            let updated_params = 
              sub params new_velocity
              |> fun p -> Constraints.apply_constraints p constraints in
            
            optimize_step updated_params new_velocity (iter + 1) divergence
          else
            params
      in
      
      let initial_velocity = zeros_like (of_float_array [|
        initial_params.cbi_params.branching.b;
        initial_params.cbi_params.branching.sigma;
        initial_params.cbi_params.immigration.beta
      |]) in
      
      optimize_step initial_params initial_velocity 0 infinity
  end
end