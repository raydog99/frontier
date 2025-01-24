open Torch

(* Domain *)
type domain =
  | WholeSpace of int  (* Rd with dimension *)
  | Torus of int       (* Td with dimension *)

let pi = 3.14159265359

(* Norm computations *)
let l1_norm tensor = 
  Tensor.sum (Tensor.abs tensor) |> Tensor.float_value

let l2_norm tensor =
  Tensor.sqrt (Tensor.sum (Tensor.pow_scalar tensor 2.)) |> Tensor.float_value

let linf_norm tensor =
  Tensor.max tensor |> Tensor.float_value

(* Numerical gradient *)  
let numerical_gradient f x epsilon =
  let shape = Tensor.shape x in
  let n = List.fold_left ( * ) 1 shape in
  let grad = Tensor.zeros shape in
  for i = 0 to n - 1 do
    let x_plus = Tensor.copy x in
    let x_minus = Tensor.copy x in
    Tensor.set x_plus [|i|] (Tensor.get x_plus [|i|] +. epsilon);
    Tensor.set x_minus [|i|] (Tensor.get x_minus [|i|] -. epsilon);
    let df = (f x_plus -. f x_minus) /. (2. *. epsilon) in
    Tensor.set grad [|i|] df
  done;
  grad

(* Laplacian operator *)
let laplacian f x h =
  let dim = List.hd (Tensor.shape x) in
  let result = Tensor.zeros_like x in
  for i = 0 to dim - 1 do
    let x_plus = Tensor.copy x in
    let x_minus = Tensor.copy x in
    Tensor.set x_plus [|i|] (Tensor.get x_plus [|i|] +. h);
    Tensor.set x_minus [|i|] (Tensor.get x_minus [|i|] -. h);
    let second_deriv = 
      (Tensor.float_value (f x_plus) +. 
       Tensor.float_value (f x_minus) -. 
       2. *. Tensor.float_value (f x)) /. (h *. h) in
    Tensor.set result [|i|] second_deriv
  done;
  result

(* Monte Carlo integration *)
let monte_carlo_integrate f domain num_samples =
  let samples = match domain with
    | WholeSpace d -> Tensor.randn [|num_samples; d|]
    | Torus d -> 
        let samples = Tensor.rand [|num_samples; d|] in
        Tensor.mul_scalar samples (2. *. pi) in
  let values = Tensor.map f samples in
  Tensor.mean values |> Tensor.float_value

(* Gauss-Legendre quadrature points and weights *)
let gauss_legendre_points n =
  let rec compute_weights m acc_points acc_weights =
    if m = 0 then (acc_points, acc_weights)
    else
      let x = cos (pi *. (float_of_int (m - 1) +. 0.75) /. 
                  (float_of_int n +. 0.5)) in
      let rec newton x0 iter =
        if iter > 100 then x0
        else
          let p0 = 1. in
          let p1 = x0 in
          let rec compute_polynomial p_prev p_curr i =
            if i > n then p_curr
            else
              let p_next = ((2. *. float_of_int i -. 1.) *. x0 *. p_curr -. 
                           (float_of_int i -. 1.) *. p_prev) /. 
                           float_of_int i in
              compute_polynomial p_curr p_next (i + 1) in
          let pn = compute_polynomial p0 p1 2 in
          let pn_prime = float_of_int n *. 
            (x0 *. pn -. p1) /. (x0 *. x0 -. 1.) in
          let x1 = x0 -. pn /. pn_prime in
          if abs_float (x1 -. x0) < 1e-15 then x1
          else newton x1 (iter + 1) in
      let x = newton x 0 in
      let w = 2. /. ((1. -. x *. x) *. 
                    (float_of_int n +. 1.) ** 2. *. 
                    (compute_polynomial 1. x 2) ** 2.) in
      compute_weights (m - 1) (x :: acc_points) (w :: acc_weights) in
  compute_weights n [] []

(* Potential module *)
module Potential = struct
  type t = {
    grad: Tensor.t -> Tensor.t;
    hessian: Tensor.t -> Tensor.t;
    is_convex: bool;
    domain: domain;
    convexity_constant: float;
  }

  let create ~grad ~hessian ~domain ~convexity_constant = 
    let is_convex = convexity_constant > 0. in
    {grad; hessian; is_convex; domain; convexity_constant}

  (* Check uniform convexity *)
  let check_uniform_convexity potential x y =
    let diff = Tensor.sub x y in
    let grad_diff = Tensor.sub (potential.grad x) (potential.grad y) in
    let inner_prod = Tensor.dot grad_diff diff in
    let norm_sq = Tensor.pow_scalar diff 2. |> Tensor.sum in
    Tensor.ge inner_prod (Tensor.mul_scalar norm_sq potential.convexity_constant)
    |> Tensor.float_value > 0.

  (* Convolution with probability density *)
  let convolve potential density x num_samples =
    let samples = Tensor.randn x.shape ~device:x.device in
    let weights = density samples in
    let convolved = Tensor.zeros_like x in
    for i = 0 to num_samples - 1 do
      let sample = Tensor.get samples [|i|] in
      let weight = Tensor.get weights [|i|] in
      let grad_eval = potential.grad (Tensor.sub x (Tensor.full_like x sample)) in
      Tensor.add_ convolved (Tensor.mul_scalar grad_eval weight)
    done;
    Tensor.div_scalar convolved (float_of_int num_samples)
end

(* Distribution and measures *)
module Distribution = struct
  type t = {
    density: Tensor.t -> Tensor.t;
    log_density: Tensor.t -> Tensor.t;
    domain: domain;
  }

  let create ~density ~log_density ~domain = {
    density;
    log_density;
    domain;
  }

  (* Relative entropy computation *)
  let relative_entropy ?(num_samples=1000) mu nu =
    let samples = Tensor.randn [|num_samples|] in
    let mu_density = mu.density samples in
    let nu_density = nu.density samples in
    let log_ratio = Tensor.sub (mu.log_density samples) (nu.log_density samples) in
    let pointwise_entropy = Tensor.mul mu_density log_ratio in
    Tensor.mean pointwise_entropy |> Tensor.float_value

  (* Relative Fisher information *)
  let relative_fisher_info ?(num_samples=1000) mu nu =
    let samples = Tensor.randn [|num_samples|] in
    let mu_density = mu.density samples in
    let grad_log_ratio = Tensor.sub 
      (Tensor.grad (mu.log_density samples) ~inputs:[samples])
      (Tensor.grad (nu.log_density samples) ~inputs:[samples]) in
    let norm_squared = Tensor.pow_scalar grad_log_ratio 2. in
    Tensor.mean (Tensor.mul mu_density norm_squared) |> Tensor.float_value
end

(* McKean SDE *)
module McKeanSDE = struct
  type t = {
    confining_potential: Potential.t;
    interaction_potential: Potential.t;
    beta: float;
    domain: domain;
    invariant_measure: Distribution.t option;
  }

  (* Create McKean SDE *)
  let create ~v ~w ~beta ~domain = {
    confining_potential = v;
    interaction_potential = w;
    beta;
    domain;
    invariant_measure = None;
  }

  (* Compute drift term *)
  let drift sde x ft =
    let v_term = Tensor.neg (sde.confining_potential.grad x) in
    let w_term = Potential.convolve sde.interaction_potential ft x 1000 in
    Tensor.sub v_term w_term

  (* Compute diffusion coefficient *)
  let diffusion sde =
    sqrt (2. /. sde.beta)

  (* Single step evolution *)
  let evolve sde xt ft dt =
    let drift_term = drift sde xt ft in
    let diff_term = diffusion sde in
    let noise = Tensor.mul_scalar (Tensor.randn_like xt) diff_term in
    Tensor.add xt (Tensor.add 
      (Tensor.mul_scalar drift_term dt)
      (Tensor.mul_scalar noise (sqrt dt)))

  (* Solve for invariant measure *)
  let solve_invariant_measure sde ~initial_guess ~max_iter ~tolerance =
    let rec iterate current_density iter =
      if iter >= max_iter then current_density
      else
        let next_density x =
          let v_term = sde.confining_potential.grad x in
          let w_term = Potential.convolve 
            sde.interaction_potential current_density x 1000 in
          let total_potential = 
            Tensor.add v_term w_term |> Tensor.mul_scalar (-1. *. sde.beta) in
          Tensor.exp total_potential in
        
        (* Normalize *)
        let normalize f =
          let z = monte_carlo_integrate f sde.domain 1000 in
          fun x -> Tensor.div (f x) (Tensor.float z) in
        
        let next_normalized = normalize next_density in
        let diff = l2_norm 
          (Tensor.sub (next_normalized (Tensor.zeros [])) 
                     (current_density (Tensor.zeros []))) in
        
        if diff < tolerance then next_normalized
        else iterate next_normalized (iter + 1) in
    
    let invariant = iterate initial_guess 0 in
    {sde with 
      invariant_measure = 
        Some {
          density = invariant;
          log_density = (fun x -> Tensor.log (invariant x));
          domain = sde.domain;
        }}

  (* Fokker-Planck equation evolution *)
  let fokker_planck sde ft dt =
    let grad_v_term x = 
      Tensor.mul (sde.confining_potential.grad x) ft in
    
    let grad_w_term x =
      let conv = Potential.convolve sde.interaction_potential ft x 1000 in
      Tensor.mul conv ft in
    
    let diffusion_term x =
      Tensor.mul_scalar (laplacian ft x 0.01) (1. /. sde.beta) in
    
    fun x ->
      let drift = Tensor.add (grad_v_term x) (grad_w_term x) in
      let diff = diffusion_term x in
      Tensor.add ft (Tensor.mul_scalar (Tensor.add drift diff) dt)
end

(* Linearized process *)
module LinearizedProcess = struct
  type t = {
    sde: McKeanSDE.t;
    invariant_measure: Distribution.t;
  }

  (* Create linearized process *)
  let create process =
    match process.McKeanSDE.invariant_measure with
    | None -> failwith "Must compute invariant measure first"
    | Some mu -> {sde = process; invariant_measure = mu}

  (* Compute linearized drift *)
  let drift process x =
    let v_term = Tensor.neg (process.sde.confining_potential.grad x) in
    let w_term = Potential.convolve 
      process.sde.interaction_potential 
      process.invariant_measure.density x 1000 in
    Tensor.sub v_term w_term

  (* Single step evolution *)
  let evolve process yt dt =
    let drift_term = drift process yt in
    let noise_scale = sqrt (2. /. process.sde.beta) in
    let noise = Tensor.mul_scalar (Tensor.randn_like yt) noise_scale in
    Tensor.add yt (Tensor.add 
      (Tensor.mul_scalar drift_term dt)
      (Tensor.mul_scalar noise (sqrt dt)))

  (* Linearized Fokker-Planck equation *)
  let fokker_planck process gt dt =
    let grad_v_term x = 
      Tensor.mul (process.sde.confining_potential.grad x) gt in
    
    let grad_w_term x =
      let conv = Potential.convolve 
        process.sde.interaction_potential 
        process.invariant_measure.density x 1000 in
      Tensor.mul conv gt in
    
    let diffusion_term x =
      Tensor.mul_scalar 
        (laplacian gt x 0.01) 
        (1. /. process.sde.beta) in
    
    fun x ->
      let drift = Tensor.add (grad_v_term x) (grad_w_term x) in
      let diff = diffusion_term x in
      Tensor.add gt (Tensor.mul_scalar (Tensor.add drift diff) dt)
end

(* Spectral analysis *)
module SpectralAnalysis = struct
  type mode = {
    index: int array;
    coefficient: Complex.t;
  }

  type spectral_decomposition = {
    modes: mode array;
    dimension: int;
    n_modes: int;
  }

  (* Create basis function *)
  let basis_function mode x =
    let term = ref Complex.one in
    Array.iter2 (fun k xi ->
      let freq = float_of_int k *. 2. *. pi in
      let basis = Complex.polar 1. (freq *. Tensor.float_value xi) in
      term := Complex.mul !term basis
    ) mode.index (Tensor.to_list1 x);
    !term

  (* Compute spectral decomposition *)
  let decompose f dim n_modes =
    let multi_indices = Array.init n_modes (fun i ->
      Array.make dim (i mod (n_modes / dim))) in
    
    let modes = Array.map (fun idx ->
      let integrand x =
        let basis = Complex.conj 
          (basis_function {index=idx; coefficient=Complex.one} x) in
        Complex.mul 
          (Complex.polar (Tensor.float_value (f x)) 0.) 
          basis in
      let coef = monte_carlo_integrate 
        (fun x -> Complex.norm (integrand x)) 
        (WholeSpace dim) 1000 in
      {index=idx; coefficient=Complex.polar coef 0.}
    ) multi_indices in
    
    {modes; dimension=dim; n_modes}

  (* Reconstruct function from decomposition *)
  let reconstruct decomp x =
    let sum = ref Complex.zero in
    Array.iter (fun mode ->
      let basis = basis_function mode x in
      sum := Complex.add !sum (Complex.mul mode.coefficient basis)
    ) decomp.modes;
    Tensor.float (Complex.norm !sum)
end

(* Fokker-Planck solver *)
module FokkerPlanck = struct
  type scheme = 
    | Explicit
    | CrankNicolson
    | ADI

  type solution = {
    times: float array;
    densities: (Tensor.t -> Tensor.t) array;
    scheme: scheme;
    dt: float;
  }

  (* Explicit scheme *)
  let evolve_explicit prev drift diffusion dt =
    fun x ->
      let drift_term = drift x prev in
      let diff_term = diffusion x prev in
      Tensor.add prev (Tensor.add 
        (Tensor.mul_scalar drift_term dt)
        (Tensor.mul_scalar diff_term dt))

  (* Crank-Nicolson scheme *)
  let evolve_crank_nicolson prev drift diffusion dt dx =
    let nx = List.hd (Tensor.shape prev) in
    let alpha = dt /. (2. *. dx *. dx) in
    
    (* Setup tridiagonal system *)
    let a = Tensor.full [|nx|] (-. alpha) in
    let b = Tensor.full [|nx|] (1. +. 2. *. alpha) in
    let c = Tensor.full [|nx|] (-. alpha) in
    
    (* Solve tridiagonal system *)
    let solve_tridiagonal a b c d =
      let n = List.hd (Tensor.shape d) in
      let x = Tensor.zeros [|n|] in
      let cp = Tensor.zeros [|n-1|] in
      let dp = Tensor.zeros [|n|] in
      
      (* Forward sweep *)
      Tensor.set cp [|0|] (Tensor.get c [|0|] /. Tensor.get b [|0|]);
      Tensor.set dp [|0|] (Tensor.get d [|0|] /. Tensor.get b [|0|]);
      
      for i = 1 to n-1 do
        let m = Tensor.get b [|i|] -. 
                Tensor.get a [|i|] *. Tensor.get cp [|i-1|] in
        if i < n-1 then
          Tensor.set cp [|i|] (Tensor.get c [|i|] /. m);
        Tensor.set dp [|i|] ((Tensor.get d [|i|] -. 
                             Tensor.get a [|i|] *. Tensor.get dp [|i-1|]) /. m)
      done;
      
      (* Back substitution *)
      Tensor.set x [|n-1|] (Tensor.get dp [|n-1|]);
      for i = n-2 downto 0 do
        Tensor.set x [|i|] (Tensor.get dp [|i|] -. 
                           Tensor.get cp [|i|] *. Tensor.get x [|i+1|])
      done;
      x in
    
    (* Right-hand side *)
    let explicit_term = 
      Tensor.add prev (Tensor.mul_scalar (drift prev) (dt /. 2.)) in
    let diffusion_term = 
      Tensor.mul_scalar (diffusion prev) (dt /. 2.) in
    let rhs = Tensor.add explicit_term diffusion_term in
    
    solve_tridiagonal a b c rhs

  (* ADI scheme *)
  let evolve_adi prev drift diffusion dt dx dy =
    let nx, ny = match Tensor.shape prev with
      | [|nx; ny|] -> nx, ny
      | _ -> failwith "Invalid tensor shape for 2D problem" in
    
    (* First half-step - implicit in x *)
    let solve_x_step prev =
      let result = Tensor.zeros [|nx; ny|] in
      for j = 0 to ny-1 do
        let line = Tensor.select prev j in
        let evolved = evolve_crank_nicolson line drift diffusion (dt/.2.) dx in
        Tensor.copy_ (Tensor.select result j) evolved
      done;
      result in
    
    (* Second half-step - implicit in y *)
    let solve_y_step intermediate =
      let result = Tensor.zeros [|nx; ny|] in
      for i = 0 to nx-1 do
        let line = Tensor.narrow intermediate 1 i 1 in
        let evolved = evolve_crank_nicolson line drift diffusion (dt/.2.) dy in
        Tensor.copy_ (Tensor.narrow result 1 i 1) evolved
      done;
      result in
    
    let intermediate = solve_x_step prev in
    solve_y_step intermediate

  (* Main solver *)
  let solve ~initial ~t_final ~dt ~scheme =
    let n_steps = int_of_float (t_final /. dt) in
    let times = Array.init n_steps (fun i -> float_of_int i *. dt) in
    let densities = Array.make n_steps initial in
    
    let evolve = match scheme with
      | Explicit -> evolve_explicit
      | CrankNicolson -> 
          fun prev drift diffusion dt ->
            evolve_crank_nicolson prev drift diffusion dt 0.01
      | ADI ->
          fun prev drift diffusion dt ->
            evolve_adi prev drift diffusion dt 0.01 0.01 in
    
    for i = 1 to n_steps - 1 do
      densities.(i) <- evolve 
        densities.(i-1)
        (fun x f -> Tensor.neg (Tensor.grad x ~inputs:[f]))
        (fun x f -> laplacian f x 0.01)
        dt
    done;
    
    {times; densities; scheme; dt}
end

(* Convergence analysis *)
module ConvergenceAnalysis = struct
  type convergence_metric = 
    | RelativeEntropy
    | L2Distance
    | WassersteinDistance

  (* LSI *)
  module LSI = struct
    type lsi_params = {
      lambda: float;
      alpha: float;
      gamma: float;
      beta: float;
      lambda0: float;
    }

    let compute_constant params t =
      let exp_term = exp (-2. *. (params.alpha +. params.gamma) *. t) in
      let uniform_constant = 
        1. /. (2. *. params.beta *. (params.alpha +. params.gamma)) in
      let lambda_t = 
        params.lambda0 *. exp_term +. uniform_constant *. (1. -. exp_term) in
      max params.lambda0 uniform_constant

    let verify measure lambda =
      Distribution.verify_lsi measure lambda
  end

  (* Compute entropy convergence *)
  let compute_entropy_convergence ~h0 ~beta ~m ~k ~alpha ~lambda t =
    if lambda < 4. /. (alpha *. beta) then
      (* Case 1 *)
      (h0 +. (beta *. beta *. m *. k) /. (4. -. alpha *. beta)) *. 
      exp (-0.5 *. alpha *. t)
    else if lambda = 4. /. (alpha *. beta) then
      (* Case 2 *)
      (h0 +. 0.5 *. beta *. m *. k *. t *. t) *. exp (-2. *. t /. beta)
    else
      (* Case 3 *)
      (h0 +. (beta *. beta *. m *. k) /. (alpha *. beta -. 4.)) *. 
      exp (-2. *. t /. beta)

  (* L2 path convergence *)
  let compute_l2_convergence ~initial_diff ~m ~k ~alpha ~gamma t =
    (initial_diff +. (8. *. m *. k) /. ((3. *. alpha +. 4. *. gamma) ** 2.)) *.
    exp (-0.5 *. alpha *. t)

  (* Wasserstein distance computation *)
  let compute_wasserstein_distance mu nu samples =
    let x_samples = mu samples in
    let y_samples = nu samples in
    let cost_matrix = Tensor.zeros [|List.hd (Tensor.shape x_samples); 
                                   List.hd (Tensor.shape y_samples)|] in
    
    for i = 0 to List.hd (Tensor.shape x_samples) - 1 do
      for j = 0 to List.hd (Tensor.shape y_samples) - 1 do
        let xi = Tensor.select x_samples i in
        let yj = Tensor.select y_samples j in
        let dist = l2_norm (Tensor.sub xi yj) in
        Tensor.set cost_matrix [|i; j|] dist
      done
    done;
    
    (* Solve optimal transport using Sinkhorn algorithm *)
    let rec sinkhorn pi eps max_iter =
      if max_iter = 0 then pi
      else
        let row_sums = Tensor.sum pi ~dim:[1] in
        let col_sums = Tensor.sum pi ~dim:[0] in
        let pi_new = Tensor.div 
          (Tensor.div pi (Tensor.unsqueeze row_sums 1))
          (Tensor.unsqueeze col_sums 0) in
        if l2_norm (Tensor.sub pi_new pi) < eps then pi_new
        else sinkhorn pi_new eps (max_iter - 1) in
    
    let transport_plan = sinkhorn 
      (Tensor.exp (Tensor.mul_scalar cost_matrix (-1.))) 
      1e-6 100 in
    
    Tensor.sum (Tensor.mul transport_plan cost_matrix) 
    |> Tensor.float_value
end

(* High-dimensional domain handling *)
module HighDimensional = struct
  (* Tensor product space *)
  type tensor_space = {
    dimensions: int array;
    total_dim: int;
    domain_type: domain;
  }

  let create_tensor_space dims domain_type = {
    dimensions = dims;
    total_dim = Array.fold_left ( * ) 1 dims;
    domain_type;
  }

  (* Multi-index operations *)
  let multi_index_to_linear dims idx =
    let rec convert i acc stride =
      if i >= Array.length dims then acc
      else convert (i + 1) 
        (acc + idx.(i) * stride) 
        (stride * dims.(i))
    in
    convert 0 0 1

  let linear_to_multi_index dims linear =
    let idx = Array.make (Array.length dims) 0 in
    let rec convert remaining i =
      if i < 0 then idx
      else begin
        idx.(i) <- remaining mod dims.(i);
        convert (remaining / dims.(i)) (i - 1)
      end
    in
    convert linear (Array.length dims - 1)

  (* Sparse grid handling *)
  type sparse_grid = {
    points: Tensor.t array;
    weights: float array;
    level: int;
  }

  let create_sparse_grid dim level =
    let n_points = int_of_float (2. ** float_of_int level) in
    let rec generate_points current_level acc =
      if current_level > level then acc
      else
        let new_points = Array.init (1 lsl current_level) (fun i ->
          let x = float_of_int i *. 2. *. pi /. 
                 float_of_int (1 lsl current_level) in
          Tensor.float x
        ) in
        generate_points (current_level + 1) (Array.append acc new_points)
    in
    let points = generate_points 0 [||] in
    let weights = Array.make (Array.length points) 
      (1. /. float_of_int (Array.length points)) in
    {points; weights; level}
end

(* Error analysis and validation *)
module ErrorAnalysis = struct
  type error_norm =
    | L1Error
    | L2Error
    | LInfError
    | RelativeError of error_norm

  (* Compute error between numerical and exact solutions *)
  let compute_error ~numerical ~exact ~norm =
    let error_tensor = Tensor.sub numerical exact in
    match norm with
    | L1Error -> l1_norm error_tensor
    | L2Error -> l2_norm error_tensor
    | LInfError -> linf_norm error_tensor
    | RelativeError base_norm ->
        let err = compute_error ~numerical ~exact ~norm:base_norm in
        let exact_norm = match base_norm with
          | L1Error -> l1_norm exact
          | L2Error -> l2_norm exact
          | LInfError -> linf_norm exact
          | RelativeError _ -> failwith "Nested relative errors not supported" in
        err /. exact_norm

  (* A posteriori error estimation *)
  let estimate_aposteriori_error ~solution ~residual ~operator ~norm =
    let res_norm = match norm with
      | L1Error -> l1_norm residual
      | L2Error -> l2_norm residual
      | LInfError -> linf_norm residual
      | RelativeError _ -> 
          failwith "Relative error not supported for residual" in
    
    (* Estimate stability constant *)
    let stability_constant = 
      SpectralAnalysis.decompose operator 1 100
      |> fun decomp -> Array.fold_left (fun acc mode ->
          max acc (Complex.norm mode.coefficient)
        ) 0. decomp.modes in
    
    res_norm *. stability_constant
end

(* Validation suite *)
module ValidationSuite = struct
  type test_case = {
    name: string;
    initial_condition: Tensor.t -> Tensor.t;
    exact_solution: float -> Tensor.t -> Tensor.t;
    domain: domain;
    t_final: float;
  }

  (* Standard test cases *)
  let standard_tests = [
    { name = "Gaussian_Diffusion";
      initial_condition = (fun x -> 
        Tensor.exp (Tensor.mul_scalar (Tensor.pow_scalar x 2.) (-1.)));
      exact_solution = (fun t x ->
        let sigma_t = sqrt (1. +. 2. *. t) in
        Tensor.div (Tensor.exp (Tensor.mul_scalar 
          (Tensor.pow_scalar x 2.) (-1. /. (2. *. sigma_t *. sigma_t))))
          (Tensor.full x.shape (sqrt (2. *. pi) *. sigma_t)));
      domain = WholeSpace 1;
      t_final = 1.0 };
    
    { name = "Periodic_Wave";
      initial_condition = (fun x ->
        Tensor.cos (Tensor.mul_scalar x (2. *. pi)));
      exact_solution = (fun t x ->
        let decay = exp (-4. *. pi *. pi *. t) in
        Tensor.mul_scalar (Tensor.cos (Tensor.mul_scalar x (2. *. pi))) 
          decay);
      domain = Torus 1;
      t_final = 0.5 }
  ]

  (* Run convergence tests *)
  let run_convergence_tests test numerical_solver =
    let dt_values = [|0.1; 0.05; 0.025; 0.0125|] in
    let errors = Array.make (Array.length dt_values) 0. in
    
    Array.iteri (fun i dt ->
      let numerical = numerical_solver 
        ~initial:test.initial_condition
        ~dt ~t_final:test.t_final in
      let exact = test.exact_solution test.t_final in
      
      errors.(i) <- ErrorAnalysis.compute_error 
        ~numerical ~exact ~norm:ErrorAnalysis.L2Error
    ) dt_values;
    
    (* Compute convergence rates *)
    let rates = Array.init (Array.length dt_values - 1) (fun i ->
      let r = log (errors.(i) /. errors.(i+1)) /. 
              log (dt_values.(i) /. dt_values.(i+1)) in
      abs_float r) in
    
    let mean_rate = Array.fold_left (+.) 0. rates /. 
                    float_of_int (Array.length rates) in
    (errors, mean_rate)
end

(* Interface *)
module McKeanLinearization = struct
  type config = {
    beta: float;
    dimension: int;
    max_iter: int;
    tolerance: float;
    t_final: float;
    dt: float;
  }

  let default_config = {
    beta = 1.0;
    dimension = 1;
    max_iter = 1000;
    tolerance = 1e-6;
    t_final = 1.0;
    dt = 0.01;
  }

  (* Create and solve complete system *)
  let solve ~confining_potential ~interaction_potential ?(config=default_config) initial =
    let sde = McKeanSDE.create 
      ~v:confining_potential 
      ~w:interaction_potential 
      ~beta:config.beta 
      ~domain:(WholeSpace config.dimension) in
    
    (* Compute invariant measure *)
    let sde_with_measure = McKeanSDE.solve_invariant_measure sde
      ~initial_guess:initial
      ~max_iter:config.max_iter
      ~tolerance:config.tolerance in
    
    (* Create linearized process *)
    let linear_process = LinearizedProcess.create sde_with_measure in
    
    (* Solve Fokker-Planck equation *)
    let solution = FokkerPlanck.solve
      ~initial
      ~t_final:config.t_final
      ~dt:config.dt
      ~scheme:FokkerPlanck.CrankNicolson in
    
    (sde_with_measure, linear_process, solution)
end