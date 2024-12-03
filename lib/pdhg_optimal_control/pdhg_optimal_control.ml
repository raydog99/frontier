open Torch

type dim1 = {
  nx: int;
  nt: int;
}

type dim2 = {
  nx: int;
  ny: int;
  nt: int;
}

type config = {
  dt: float;
  dx: float;
  dy: float option;
  epsilon: float;
  tau_rho: float;
  tau_alpha: float;
  tau_phi: float;
  max_iter: int;
  tolerance: float;
}

let d_plus_x tensor dx =
  div (sub (slice tensor [Some 1; None]) 
          (slice tensor [Some 0; None]))
      (Scalar.f dx)

let d_minus_x tensor dx =
  div (sub (slice tensor [Some 0; None])
          (slice tensor [Some (-1); None]))
      (Scalar.f dx)

let laplacian tensor dx =
  let dx2 = dx *. dx in
  div (add (sub (add (slice tensor [Some 1; None])
                    (slice tensor [Some (-1); None]))
               (mul (Scalar.f 2.) (slice tensor [Some 0; None])))
          (zeros_like tensor))
      (Scalar.f dx2)

let d_time tensor dt =
  div (sub (slice tensor [None; Some 1])
          (slice tensor [None; Some 0]))
      (Scalar.f dt)

let gradient tensor =
  grad tensor tensor

let adjoint tensor =
  transpose tensor 0 1

let second_derivative_x tensor dx =
  let dx2 = dx *. dx in
  div (add (sub (add (slice tensor [Some 2; None])
                    (slice tensor [Some 0; None]))
               (mul (Scalar.f 2.) 
                    (slice tensor [Some 1; None])))
          (zeros_like tensor))
      (Scalar.f dx2)

let second_derivative_y tensor dy =
  let dy2 = dy *. dy in
  div (add (sub (add (slice tensor [None; Some 2])
                    (slice tensor [None; Some 0]))
               (mul (Scalar.f 2.) 
                    (slice tensor [None; Some 1])))
          (zeros_like tensor))
      (Scalar.f dy2)

let cross_derivative_xy tensor dx dy =
  let dxdy = dx *. dy in
  div (sub (sub (add (slice tensor [Some 1; Some 1])
                    (slice tensor [Some (-1); Some (-1)]))
               (slice tensor [Some 1; Some (-1)]))
          (slice tensor [Some (-1); Some 1]))
      (Scalar.f (4. *. dxdy))

module Domain = struct
  type t = {
    x_min: float;
    x_max: float;
    y_min: float option;
    y_max: float option;
    t_min: float;
    t_max: float;
  }

  let create_1d ~x_min ~x_max ~t_min ~t_max = {
    x_min; x_max;
    y_min = None; y_max = None;
    t_min; t_max
  }

  let create_2d ~x_min ~x_max ~y_min ~y_max ~t_min ~t_max = {
    x_min; x_max;
    y_min = Some y_min;
    y_max = Some y_max;
    t_min; t_max
  }
end

module OptimalControl = struct
  type state = {
    phi: Tensor.t;
    rho: Tensor.t;
    alpha: Tensor.t;
    phi_tilde: Tensor.t;
  }

  let create_state ~nx ~nt ~initial =
    let shape = [nx; nt] in
    {
      phi = initial;
      rho = Tensor.zeros shape;
      alpha = Tensor.zeros shape;
      phi_tilde = initial;
    }

  module PDHG = struct
    let step config state =
      (* Update rho *)
      let d_phi = d_time state.phi_tilde config.dt in
      let grad_phi = gradient state.phi_tilde in
      let rho_update = add state.rho 
        (mul (Scalar.f config.tau_rho) 
             (add d_phi (neg (mul state.alpha grad_phi)))) in
      let new_rho = max rho_update (zeros_like rho_update) in
      
      (* Update alpha *)
      let alpha_update = add state.alpha
        (mul (Scalar.f config.tau_alpha)
             (mul new_rho grad_phi)) in
      let new_alpha = alpha_update in
      
      (* Update phi *)
      let d_rho = d_time new_rho config.dt in
      let div_alpha = adjoint new_alpha in
      let phi_update = add state.phi
        (mul (Scalar.f config.tau_phi)
             (add d_rho (neg div_alpha))) in
      let new_phi = phi_update in
      
      (* Update phi_tilde *)
      let phi_tilde = add (mul (Scalar.f 2.) new_phi)
                         (neg state.phi) in
      
      { phi = new_phi;
        rho = new_rho;
        alpha = new_alpha;
        phi_tilde }

    let solve ?(max_iter=1000) ?(tol=1e-6) config state =
      let rec iterate state n =
        if n >= max_iter then state
        else
          let next = step config state in
          let diff = Tensor.(sum (abs (sub next.phi state.phi))) in
          if Tensor.float_value diff < tol then next
          else iterate next (n + 1)
      in
      iterate state 0
  end
end

module CompleteSaddlePoint = struct
  type density_state = {
    rho: Tensor.t;
    active_set: Tensor.t;
    inactive_set: Tensor.t;
  }

  let update_density_state ~rho ~epsilon =
    let active = gt rho (Scalar.f epsilon) in
    let inactive = logical_not active in
    { rho; active_set = active; inactive_set = inactive }

  let modified_pdhg_step ~state ~config =    
    (* Process active set *)
    let active_rho = where state.active_set 
                          state.rho 
                          (zeros_like state.rho) in
    
    (* Compute updates on active set *)
    let phi_active = where state.active_set 
                          state.rho
                          (zeros_like state.rho) in
    
    (* Standard PDHG updates on active regions *)
    let d_phi = d_time phi_active config.dt in
    let grad_phi = gradient phi_active in
    
    let rho_update = add active_rho 
      (mul (Scalar.f config.tau_rho) 
           (add d_phi (neg (mul active_rho grad_phi)))) in
    
    (* Project back onto feasible set *)
    let rho_projected = max rho_update (zeros_like rho_update) in
    
    (* Final result *)
    where state.active_set rho_projected (zeros_like rho_projected)
end

module CompletePhaseSpace = struct
  type phase_state = {
    position: Tensor.t;
    velocity: Tensor.t;
    momentum: Tensor.t;
    time: float;
  }

  let symplectic_evolve ~state ~dt ~force =
    (* Half step in position *)
    let half_dt = dt /. 2. in
    let mid_pos = add state.position 
                     (mul (Scalar.f half_dt) state.velocity) in
    
    (* Full step in velocity using force at midpoint *)
    let force_mid = force mid_pos state.velocity state.time in
    let new_vel = add state.velocity 
                     (mul (Scalar.f dt) force_mid) in
    
    (* Half step in position using new velocity *)
    let new_pos = add mid_pos 
                     (mul (Scalar.f half_dt) new_vel) in
    
    (* Update momentum *)
    let new_mom = mul (Scalar.f state.time) new_vel in
    
    { position = new_pos;
      velocity = new_vel;
      momentum = new_mom;
      time = state.time +. dt }

  let solve ~initial ~config =
    let rec evolve state n =
      if n >= config.max_iter then state
      else
        let next = symplectic_evolve 
          ~state 
          ~dt:config.dt
          ~force:(fun x v t -> Tensor.zeros_like x) in
        
        (* Check convergence *)
        let diff = Tensor.(sum (abs (sub next.position state.position))) in
        if Tensor.float_value diff < config.tolerance then next
        else evolve next (n + 1)
    in
    evolve initial 0
end

module CompleteFokkerPlanck = struct
  type fp_state = {
    density: Tensor.t;
    drift: Tensor.t;
    diffusion: Tensor.t;
    time: float;
  }

  let solve_fokker_planck ~initial ~drift ~diffusion ~config =
    let step state =
      (* Compute spatial derivatives *)
      let dx = d_plus_x state.density config.dx in
      let dxx = second_derivative_x state.density config.dx in
      
      (* Compute drift and diffusion updates *)
      let drift_term = mul (drift state.density state.time) dx in
      let diff_term = mul (diffusion state.density state.time) dxx in
      
      (* Forward Euler step *)
      let new_density = add state.density 
        (mul (Scalar.f config.dt) 
             (add (neg drift_term) 
                  (mul (Scalar.f 0.5) diff_term))) in
      
      (* Normalize to preserve mass *)
      let total_mass = sum new_density in
      let normalized = div new_density total_mass in
      
      { density = normalized;
        drift = state.drift;
        diffusion = state.diffusion;
        time = state.time +. config.dt }
    in
    
    let rec evolve state n =
      if n >= config.max_iter then state
      else
        let next = step state in
        (* Check convergence *)
        let diff = Tensor.(sum (abs (sub next.density state.density))) in
        if Tensor.float_value diff < config.tolerance then next
        else evolve next (n + 1)
    in
    
    evolve {
      density = initial;
      drift = zeros_like initial;
      diffusion = zeros_like initial;
      time = 0.
    } 0
end

module CompleteMixedBoundary = struct
  type boundary_type =
    | Periodic
    | Neumann
    | Dirichlet of float
    | Mixed of boundary_type * boundary_type

  let apply_mixed_boundary ~state ~bc_type ~config =
    match bc_type with
    | Mixed (pos_bc, vel_bc) ->
        let handle_periodic tensor =
          let n = (size tensor 0) - 1 in
          copy_ ~src:(slice tensor [Some 0; None]) 
                ~dst:(slice tensor [Some n; None]);
          copy_ ~src:(slice tensor [Some (n-1); None]) 
                ~dst:(slice tensor [Some (-1); None]);
          tensor
        in
        
        let handle_neumann tensor =
          let n = (size tensor 0) - 1 in
          (* Zero gradient at boundaries *)
          copy_ ~src:(slice tensor [Some 1; None]) 
                ~dst:(slice tensor [Some 0; None]);
          copy_ ~src:(slice tensor [Some (n-1); None]) 
                ~dst:(slice tensor [Some n; None]);
          tensor
        in
        
        let handle_dirichlet tensor value =
          let n = (size tensor 0) - 1 in
          let bound = full_like (slice tensor [Some 0; None]) value in
          copy_ ~src:bound ~dst:(slice tensor [Some 0; None]);
          copy_ ~src:bound ~dst:(slice tensor [Some n; None]);
          tensor
        in
        
        let new_phi = match pos_bc with
          | Periodic -> handle_periodic state.phi
          | Neumann -> handle_neumann state.phi
          | Dirichlet v -> handle_dirichlet state.phi v
          | Mixed _ -> failwith "Nested mixed conditions not supported"
        in
        
        let new_rho = match vel_bc with
          | Periodic -> handle_periodic state.rho
          | Neumann -> handle_neumann state.rho
          | Dirichlet v -> handle_dirichlet state.rho v
          | Mixed _ -> failwith "Nested mixed conditions not supported"
        in
        
        { state with 
          phi = new_phi; 
          rho = new_rho;
          phi_tilde = new_phi }
    | _ -> failwith "Expected mixed boundary conditions"
end

module NewtonMechanics = struct
  type state = {
    position: Tensor.t;
    velocity: Tensor.t;
    momentum: Tensor.t;
    time: float;
  }

  let create_state ~initial_pos ~initial_vel ~time =
    {
      position = initial_pos;
      velocity = initial_vel;
      momentum = mul initial_vel (Scalar.f time);
      time;
    }

  let solve_newton_mechanics ~initial ~force ~config =
    let step state =
      (* Update position using velocity *)
      let new_pos = add state.position 
        (mul (Scalar.f config.dt) state.velocity) in
      
      (* Update velocity using force *)
      let new_vel = add state.velocity 
        (mul (Scalar.f config.dt) force) in
      
      (* Update momentum *)
      let new_mom = mul (Scalar.f state.time) new_vel in
      
      { position = new_pos;
        velocity = new_vel;
        momentum = new_mom;
        time = state.time +. config.dt }
    in
    
    let rec evolve state n =
      if n >= config.max_iter then state
      else evolve (step state) (n + 1)
    in
    evolve initial 0
end

module StochasticControl = struct
  type stochastic_state = {
    value: Tensor.t;
    noise: Tensor.t;
    paths: Tensor.t list;
    time: float;
  }

  let create_stochastic_state ~initial ~time =
    {
      value = initial;
      noise = Tensor.zeros_like initial;
      paths = [initial];
      time;
    }

  let solve_viscous_hjb ~hamiltonian ~epsilon ~config ~initial =    
    let generate_noise shape =
      let dw = randn shape in
      mul dw (sqrt (Scalar.f config.dt))
    in
    
    let step state =
      (* Generate noise *)
      let noise = generate_noise (size state.value) in
      
      (* Compute drift term *)
      let drift = neg hamiltonian in
      
      (* Compute diffusion term *)
      let diff_term = mul (Scalar.f epsilon) 
        (laplacian state.value config.dx) in
      
      (* Update value *)
      let new_value = add state.value
        (add (mul (Scalar.f config.dt) (add drift diff_term))
             (mul (sqrt (Scalar.f (2. *. epsilon))) noise)) in
      
      { value = new_value;
        noise;
        paths = new_value :: state.paths;
        time = state.time +. config.dt }
    in
    
    let rec evolve state n =
      if n >= config.max_iter then state
      else evolve (step state) (n + 1)
    in
    evolve initial 0
end

module TimeDependent = struct
  type coefficient = {
    spatial: Tensor.t -> float -> Tensor.t;
    temporal: float -> Tensor.t -> Tensor.t;
    mixed: Tensor.t -> float -> Tensor.t;
  }

  let eval_coefficients ~x ~t coeffs =
    let space_coeff = coeffs.spatial x t in
    let time_coeff = coeffs.temporal t x in
    let mixed_coeff = coeffs.mixed x t in
    { space_coeff; time_coeff; mixed_coeff }
end

module ViscousTreatment = struct
  type viscous_params = {
    epsilon: float;
    anisotropic: bool;
    dimension_coeffs: float array;
  }

  let discretize_viscous_terms ~phi ~params ~config =
    (* Handle each dimension *)
    let handle_dimension dim =
      let coeff = params.dimension_coeffs.(dim) in
      match dim with
      | 0 -> (* x direction *)
          let d2x = second_derivative_x phi config.dx in
          mul (Scalar.f (params.epsilon *. coeff)) d2x
      | 1 -> (* y direction *)
          let d2y = second_derivative_y phi 
            (Option.value ~default:config.dx config.dy) in
          mul (Scalar.f (params.epsilon *. coeff)) d2y
      | _ -> zeros_like phi
    in
    
    (* Combine all dimensional contributions *)
    let dims = Array.length params.dimension_coeffs in
    let result = ref (zeros_like phi) in
    for d = 0 to dims - 1 do
      result := add !result (handle_dimension d)
    done;
    !result

  let apply_viscous_boundary ~state ~params ~config =
    (* Apply boundary conditions for each dimension *)
    let apply_dim_boundary dim tensor =
      let n = (size tensor 0) - 1 in
      let coeff = params.dimension_coeffs.(dim) in
      let boundary_value = params.epsilon *. coeff in
      
      let bound = full_like (slice tensor [Some 0; None]) boundary_value in
      copy_ ~src:bound ~dst:(slice tensor [Some 0; None]);
      copy_ ~src:bound ~dst:(slice tensor [Some n; None]);
      tensor
    in
    
    let new_phi = Array.fold_left
      (fun acc dim -> apply_dim_boundary dim acc)
      state.phi
      (Array.init (Array.length params.dimension_coeffs) Fun.id)
    in
    
    { state with phi = new_phi }
end