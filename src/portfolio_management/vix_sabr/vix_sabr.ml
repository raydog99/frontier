open Torch

type parameters = {
  beta: float;
  omega: float;  (* volatility of volatility *)
  rho: float;    (* correlation *)
  alpha: float;  (* = (1-beta)/2 *)
  delta: float;  (* = -(1-beta)ρω *)
}

type state = {
  s: float Tensor.t;  (* Asset price *)
  v: float Tensor.t;  (* Volatility *)
  t: float;          (* Current time *)
}

type simulation_config = {
  n_paths: int;
  n_steps: int;
  dt: float;
  use_brownian_bridge: bool;
  variance_reduction: bool;
  antithetic: bool;
}

type integration_config = {
  rel_tol: float;
  abs_tol: float;
  max_iter: int;
  method_name: string;
}

type boundary_type =
  | Regular     (* Accessible and can exit *)
  | Exit       (* Accessible but cannot return *)
  | Entrance   (* Inaccessible but can enter *)
  | Natural    (* Inaccessible and cannot enter *)

type boundary_classification = {
  zero_type: boundary_type;
  infinity_type: boundary_type;
}

type solver_config = {
  rel_tol: float;
  abs_tol: float;
  max_steps: int;
  stability_factor: float;
}

type 'a integration_result = {
  value: 'a;
  error_est: float;
  n_steps: int;
}

type time_span = {
  t0: float;
  t1: float;
}

(* Adaptive quadrature integration *)
let integrate ?(config={rel_tol=1e-8; abs_tol=1e-10; max_iter=1000; method_name="adaptive"}) f a b =
  let open Float in
  
  let rec adaptive_quad f a b fa fm fb tol depth =
    if depth > config.max_iter then
      Error "Maximum iterations exceeded"
    else
      let m = (a +. b) /. 2.0 in
      let h = (b -. a) /. 6.0 in
      let f1 = f ((2.0 *. a +. b) /. 3.0) in
      let f2 = f ((a +. 2.0 *. b) /. 3.0) in
      let s1 = h *. (fa +. 4.0 *. fm +. fb) in
      let s2 = h *. (fa +. 4.0 *. f1 +. 2.0 *. fm +. 4.0 *. f2 +. fb) in
      
      if abs_float (s1 -. s2) < 15.0 *. tol then
        Ok { 
          value = s2;
          error_est = abs_float (s1 -. s2) /. 15.0;
          n_steps = depth 
        }
      else
        match adaptive_quad f a m fa f1 fm (tol /. 2.0) (depth + 1) with
        | Error e -> Error e
        | Ok left ->
            match adaptive_quad f m b fm f2 fb (tol /. 2.0) (depth + 1) with
            | Error e -> Error e
            | Ok right ->
                Ok {
                  value = left.value +. right.value;
                  error_est = left.error_est +. right.error_est;
                  n_steps = left.n_steps + right.n_steps
                }
  in
  
  let fa = f a in
  let fm = f ((a +. b) /. 2.0) in
  let fb = f b in
  adaptive_quad f a b fa fm fb config.rel_tol 0

(* Core SABR model *)
module Sabr = struct
  let create_parameters ~beta ~omega ~rho =
    let alpha = (1.0 -. beta) /. 2.0 in
    let delta = -.(1.0 -. beta) *. rho *. omega in
    { beta; omega; rho; alpha; delta }

  (* Compute σ_V(v) *)
  let sigma_v params v =
    let open Tensor in
    let beta_m1 = params.beta -. 1.0 in
    let term1 = params.omega ** (float_vec 2.0) in
    let term2 = (beta_m1 ** (float_vec 2.0)) * (v ** (float_vec 2.0)) in
    let term3 = float_vec (2.0 *. params.rho *. beta_m1 *. params.omega) * v in
    sqrt((float_vec term1 + term2 + term3))

  (* Compute μ_V(v) *)
  let mu_v params v =
    let open Tensor in
    let beta_m1 = params.beta -. 1.0 in
    let term1 = beta_m1 *. (0.5 *. (params.beta -. 2.0)) in
    let term2 = params.rho *. params.omega in
    v * (float_vec (term1 +. term2))

  (* Evolution step using Euler-Maruyama scheme *)
  let evolve_step params state dt =
    let open Tensor in
    let dw = randn_like state.v in
    let drift = mu_v params state.v * (float_vec dt) in
    let diffusion = (sigma_v params state.v * dw) * (float_vec (sqrt dt)) in
    let new_v = state.v + drift + diffusion in
    
    (* Update asset price *)
    let db = randn_like state.s in
    let corr_term = params.rho *. dt |> sqrt |> float_vec in
    let db = db * corr_term + dw * (float_vec (sqrt (1.0 -. params.rho ** 2.0) *. dt)) in
    let new_s = state.s * (float_vec (1.0 +. params.beta)) * new_v * db in
    
    { s = new_s; v = new_v; t = state.t +. dt }

  (* Generate full path *)
  let simulate params initial_state config =
    let dt = config.dt in
    let rec loop state acc steps =
      if steps >= config.n_steps then List.rev (state :: acc)
      else
        let next_state = evolve_step params state dt in
        loop next_state (state :: acc) (steps + 1)
    in
    loop initial_state [] 0

  let validate_parameters params =
    let open Float in
    let valid_beta = params.beta >= 0.0 && params.beta < 1.0 in
    let valid_rho = abs_float params.rho <= 1.0 in
    let valid_omega = params.omega > 0.0 in
    
    match (valid_beta, valid_rho, valid_omega) with
    | false, _, _ -> Error "Invalid beta: must be in [0,1)"
    | _, false, _ -> Error "Invalid rho: must be in [-1,1]"
    | _, _, false -> Error "Invalid omega: must be positive"
    | true, true, true -> Ok params
end

(* R(x) function  *)
let calc_R params x =
  let open Tensor in
  let beta_m1 = params.beta -. 1.0 in
  let term1 = params.omega ** (float_vec 2.0) in
  let term2 = float_vec (2.0 *. params.omega *. params.rho *. beta_m1) * x in
  let term3 = float_vec (beta_m1 ** 2.0) * (x ** (float_vec 2.0)) in
  term1 + term2 + term3

(* F(x) function *)
let calc_F params x =
  let open Tensor in
  let beta_m1 = params.beta -. 1.0 in
  let rho_perp = sqrt (1.0 -. params.rho ** 2.0) in
  let omega_bar = params.omega *. abs_float params.rho in
  
  let r = calc_R params x in
  let term1 = float_vec (params.alpha /. (beta_m1 ** 2.0)) * 
              log (r / (float_vec (params.omega ** 2.0))) in
  
  let arg1 = (float_vec beta_m1 * x + float_vec omega_bar) / 
             (float_vec (params.omega *. rho_perp)) in
  let arg2 = float_vec (omega_bar /. (params.omega *. rho_perp)) in
  
  let term2 = float_vec (-0.5 *. beta_m1 *. (2.0 -. params.beta) /. 
              params.rho *. rho_perp) * 
              (atan2 arg1 (float_vec 1.0) - atan2 arg2 (float_vec 1.0)) in
  
  term1 + term2

(* Enhanced scale function with proper boundary conditions *)
let calc_scale_function params =
  let open Tensor in
  let integrand y =
    let f = calc_F params y in
    exp (float_vec (-2.0) * f)
  in
  
  (* Gaussian quadrature integration *)
  let gauss_legendre_points n =
    let open Float in
    let rec legendre_poly x n =
      if n = 0 then 1.0
      else if n = 1 then x
      else
        let p0 = legendre_poly x (n-2) in
        let p1 = legendre_poly x (n-1) in
        ((2.0 *. float_of_int (n-1) +. 1.0) *. x *. p1 -.
         float_of_int (n-1) *. p0) /. float_of_int n
    in
    
    let rec newton_root x n iter =
      if iter > 50 then x
      else
        let p = legendre_poly x n in
        let dp = float_of_int n *. 
                (x *. legendre_poly x n -. legendre_poly x (n-1)) /.
                (x *. x -. 1.0) in
        let x_new = x -. p /. dp in
        if abs_float (x_new -. x) < 1e-15 then x_new
        else newton_root x_new n (iter + 1)
    in
    
    let x0 = Array.init n (fun i ->
      cos (pi *. float_of_int (2*i + 1) /. float_of_int (2*n))
    ) in
    Array.mapi (fun i x -> newton_root x n 0) x0
  in
  
  let gauss_legendre_weights nodes n =
    Array.map (fun x ->
      2.0 /. (1.0 -. x *. x)
    ) nodes
  in
  
  fun x ->
    let n = 50 in (* number of quadrature points *)
    let nodes = gauss_legendre_points n in
    let weights = gauss_legendre_weights nodes n in
    
    let sum = ref (float_vec 0.0) in
    for i = 0 to n - 1 do
      let y = float_of_int i *. x /. float_of_int n in
      sum := !sum + float_vec weights.(i) * integrand (float_vec y)
    done;
    to_float0_exn (!sum * float_vec (x /. 2.0))

(* Milstein scheme implementation *)
module Milstein = struct
  type scheme_config = {
    use_antithetic: bool;
    correction_term: bool;
    n_paths: int;
    dt: float;
  }

  (* Helper for derivative calculations *)
  let calc_derivatives params v =
    let open Tensor in
    let h = 1e-6 in
    let v_plus = v + float_vec h in
    let v_minus = v - float_vec h in
    
    let sigma_plus = Sabr.sigma_v params v_plus in
    let sigma_minus = Sabr.sigma_v params v_minus in
    let dsigma_dv = (sigma_plus - sigma_minus) / (float_vec (2.0 *. h)) in
    
    let mu_plus = Sabr.mu_v params v_plus in
    let mu_minus = Sabr.mu_v params v_minus in
    let dmu_dv = (mu_plus - mu_minus) / (float_vec (2.0 *. h)) in
    
    (dsigma_dv, dmu_dv)

  (* Single step of Milstein scheme *)
  let step params state config dt =
    let open Tensor in
    let v = state.v in
    let s = state.s in
    
    (* Generate correlated Brownian increments *)
    let z1 = randn [config.n_paths] in
    let z2 = randn [config.n_paths] in
    let dw = z1 in
    let dz = float_vec params.rho * z1 + 
             float_vec (sqrt (1.0 -. params.rho *. params.rho)) * z2 in
    
    (* Calculate coefficients *)
    let sigma = Sabr.sigma_v params v in
    let mu = Sabr.mu_v params v in
    let dsigma_dv, dmu_dv = calc_derivatives params v in
    
    (* Milstein correction terms *)
    let correction = 
      if config.correction_term then
        float_vec 0.5 * sigma * dsigma_dv * 
        (dw * dw - float_vec dt) / float_vec (sqrt dt)
      else float_vec 0.0 in
    
    (* Update volatility *)
    let v_new = v + mu * float_vec dt + sigma * dw * float_vec (sqrt dt) + 
               correction in
    
    (* Update asset price *)
    let s_new = s * exp(
      float_vec params.beta * v * dz * float_vec (sqrt dt) -
      float_vec 0.5 * (v * v) * float_vec dt
    ) in
    
    { state with v = v_new; s = s_new; t = state.t +. dt }

  (* Generate full path with Milstein scheme *)
  let simulate params initial_state config n_steps =
    let dt = config.dt in
    let rec loop state acc steps =
      if steps >= n_steps then List.rev (state :: acc)
      else
        let next_state = step params state config dt in
        loop next_state (state :: acc) (steps + 1)
    in
    loop initial_state [] 0
end

(* Wagner-Platen scheme for higher order accuracy *)
module WagnerPlaten = struct
  let step params state dt =
    let open Tensor in
    let v = state.v in
    let s = state.s in
    
    (* Generate Brownian increments *)
    let dw = randn_like v in
    let dz = randn_like v in
    
    (* First order terms *)
    let mu = Sabr.mu_v params v in
    let sigma = Sabr.sigma_v params v in
    
    (* Second order terms *)
    let dsigma_dv, dmu_dv = Milstein.calc_derivatives params v in
    
    (* Higher order corrections *)
    let correction1 = mu * dmu_dv * float_vec (dt ** 2.0) / float_vec 2.0 in
    let correction2 = sigma * dsigma_dv * 
      (dw * dw - float_vec dt) / float_vec 2.0 in
    let correction3 = sigma * mu * float_vec dt * dw in
    
    (* Update volatility *)
    let v_new = v + mu * float_vec dt + sigma * dw * float_vec (sqrt dt) +
               correction1 + correction2 + correction3 in
               
    (* Update asset price with corresponding terms *)
    let s_new = s * exp(
      float_vec params.beta * v * dz * float_vec (sqrt dt) -
      float_vec 0.5 * (v * v) * float_vec dt +
      float_vec params.beta * correction2 * float_vec (dt / sqrt dt)
    ) in
    
    { state with v = v_new; s = s_new; t = state.t +. dt }

  (* Generate full path with Wagner-Platen scheme *)
  let simulate params initial_state dt n_steps =
    let rec loop state acc steps =
      if steps >= n_steps then List.rev (state :: acc)
      else
        let next_state = step params state dt in
        loop next_state (state :: acc) (steps + 1)
    in
    loop initial_state [] 0
end

(* VIX *)
module Vix = struct
  (* Calculate VIX *)
  let calc_vix params state tau =
    let open Tensor in
    let v = state.v in
    let s = state.s in
    let v_effective = s ** (float_vec (params.beta -. 1.0)) * v in
    sqrt (v_effective * v_effective) |> mean |> to_float0_exn

  (* VIX option pricing and analysis *)
  module Options = struct
    type pricing_config = {
      n_paths: int;
      dt: float;
      variance_reduction: bool;
      error_est: bool;
    }

    type pricing_result = {
      price: float;
      error: float option;
      greeks: greeks option;
      implied_vol: float option;
    }
    and greeks = {
      delta: float;
      gamma: float;
      vega: float;
      theta: float;
      rho: float;
    }

    (* Black-Scholes formula for VIX options *)
    let black_scholes ~s ~k ~r ~t ~sigma =
      let open Float in
      let d1 = (log (s /. k) +. (r +. 0.5 *. sigma *. sigma) *. t) /. 
               (sigma *. sqrt t) in
      let d2 = d1 -. sigma *. sqrt t in
      
      let nd1 = Torch.Special.ndtr (Tensor.float_vec d1) |> Tensor.to_float0_exn in
      let nd2 = Torch.Special.ndtr (Tensor.float_vec d2) |> Tensor.to_float0_exn in
      let npd1 = exp (-0.5 *. d1 *. d1) /. sqrt (2.0 *. pi) in
      
      (* Option prices *)
      let call = s *. nd1 -. k *. exp (-.r *. t) *. nd2 in
      let put = k *. exp (-.r *. t) *. (1.0 -. nd2) -. s *. (1.0 -. nd1) in
      
      (* Greeks *)
      let delta = nd1 in
      let gamma = npd1 /. (s *. sigma *. sqrt t) in
      let vega = s *. sqrt t *. npd1 in
      let theta = -.(s *. sigma *. npd1) /. (2.0 *. sqrt t) -. 
                 r *. k *. exp (-.r *. t) *. nd2 in
      let rho = k *. t *. exp (-.r *. t) *. nd2 in
      
      ( (call, put), 
        { delta; gamma; vega; theta; rho }
      )

    (* Monte Carlo pricing of VIX call option *)
    let price_call params state ~strike ~maturity config =
      let paths = StochasticIntegration.Milstein.simulate params state {
        use_antithetic = config.variance_reduction;
        correction_term = true;
        n_paths = config.n_paths;
        dt = config.dt
      } (int_of_float (maturity /. config.dt)) in
      
      (* Calculate payoffs *)
      let payoffs = List.map (fun path ->
        let terminal_vix = calc_vix params (List.hd (List.rev path)) config.dt in
        max (terminal_vix -. strike) 0.0
      ) paths in
      
      (* Calculate price and error estimate *)
      let mean_payoff = List.fold_left (+.) 0.0 payoffs /. 
                       float_of_int (List.length payoffs) in
      
      let error = 
        if config.error_est then
          let variance = List.fold_left (fun acc x ->
            acc +. (x -. mean_payoff) ** 2.0
          ) 0.0 payoffs in
          let std_dev = sqrt (variance /. float_of_int (List.length payoffs - 1)) in
          Some (1.96 *. std_dev /. sqrt (float_of_int config.n_paths))
        else None in
      
      (* Calculate Greeks using finite differences *)
      let greeks =
        let delta_bump = state.v *. 0.01 in
        let up_state = { state with v = Tensor.(state.v + float_vec delta_bump) } in
        let down_state = { state with v = Tensor.(state.v - float_vec delta_bump) } in
        
        let up_price = price_call params up_state ~strike ~maturity 
          {config with error_est=false} |> fst in
        let down_price = price_call params down_state ~strike ~maturity 
          {config with error_est=false} |> fst in
        
        let delta = (up_price -. down_price) /. (2.0 *. delta_bump) in
        let gamma = (up_price -. 2.0 *. mean_payoff +. down_price) /. 
                   (delta_bump *. delta_bump) in
        
        (* Vega calculation *)
        let omega_bump = params.omega *. 0.01 in
        let up_params = {params with omega = params.omega +. omega_bump} in
        let vega_price = price_call up_params state ~strike ~maturity 
          {config with error_est=false} |> fst in
        let vega = (vega_price -. mean_payoff) /. omega_bump in
        
        (* Theta approximation *)
        let dt = config.dt /. 2.0 in
        let future_price = price_call params state ~strike ~maturity:(maturity +. dt)
          {config with error_est=false} |> fst in
        let theta = (future_price -. mean_payoff) /. dt in
        
        Some {delta; gamma; vega; theta; rho = 0.0}  
      in
      
      (* Compute implied volatility *)
      let implied_vol = 
        let spot_vix = calc_vix params state config.dt in
        let rec newton_vol sigma iter =
          if iter > 50 then None
          else
            let (bs_price, _), _ = black_scholes 
              ~s:spot_vix ~k:strike ~r:0.0 ~t:maturity ~sigma in
            let diff = bs_price -. mean_payoff in
            if abs_float diff < 1e-8 then Some sigma
            else
              let ((up_price, _), _) = black_scholes 
                ~s:spot_vix ~k:strike ~r:0.0 ~t:maturity ~sigma:(sigma *. 1.01) in
              let vega = (up_price -. bs_price) /. (0.01 *. sigma) in
              if abs_float vega < 1e-10 then None
              else
                let new_sigma = sigma -. diff /. vega in
                if new_sigma <= 0.0 then None
                else newton_vol new_sigma (iter + 1)
        in
        newton_vol 0.5 0
      in
      
      mean_payoff, 
      { price = mean_payoff;
        error;
        greeks;
        implied_vol }

    (* Monte Carlo pricing of VIX put option *)
    let price_put params state ~strike ~maturity config =
      let paths = StochasticIntegration.Milstein.simulate params state {
        use_antithetic = config.variance_reduction;
        correction_term = true;
        n_paths = config.n_paths;
        dt = config.dt
      } (int_of_float (maturity /. config.dt)) in
      
      let payoffs = List.map (fun path ->
        let terminal_vix = calc_vix params (List.hd (List.rev path)) config.dt in
        max (strike -. terminal_vix) 0.0
      ) paths in
      
      let mean_payoff = List.fold_left (+.) 0.0 payoffs /. 
                       float_of_int (List.length payoffs) in
      
      let error =
        if config.error_est then
          let variance = List.fold_left (fun acc x ->
            acc +. (x -. mean_payoff) ** 2.0
          ) 0.0 payoffs in
          let std_dev = sqrt (variance /. float_of_int (List.length payoffs - 1)) in
          Some (1.96 *. std_dev /. sqrt (float_of_int config.n_paths))
        else None in
      
      (* Greeks calculation similar to calls *)
      let greeks =
        let delta_bump = state.v *. 0.01 in
        let up_state = { state with v = Tensor.(state.v + float_vec delta_bump) } in
        let down_state = { state with v = Tensor.(state.v - float_vec delta_bump) } in
        
        let up_price = price_put params up_state ~strike ~maturity 
          {config with error_est=false} |> fst in
        let down_price = price_put params down_state ~strike ~maturity 
          {config with error_est=false} |> fst in
        
        let delta = (up_price -. down_price) /. (2.0 *. delta_bump) in
        let gamma = (up_price -. 2.0 *. mean_payoff +. down_price) /. 
                   (delta_bump *. delta_bump) in
        
        let omega_bump = params.omega *. 0.01 in
        let up_params = {params with omega = params.omega +. omega_bump} in
        let vega_price = price_put up_params state ~strike ~maturity 
          {config with error_est=false} |> fst in
        let vega = (vega_price -. mean_payoff) /. omega_bump in
        
        let dt = config.dt /. 2.0 in
        let future_price = price_put params state ~strike ~maturity:(maturity +. dt)
          {config with error_est=false} |> fst in
        let theta = (future_price -. mean_payoff) /. dt in
        
        Some {delta; gamma; vega; theta; rho = 0.0}
      in
      
      (* Compute implied volatility *)
      let implied_vol =
        let spot_vix = calc_vix params state config.dt in
        let rec newton_vol sigma iter =
          if iter > 50 then None
          else
            let (_, bs_price), _ = black_scholes 
              ~s:spot_vix ~k:strike ~r:0.0 ~t:maturity ~sigma in
            let diff = bs_price -. mean_payoff in
            if abs_float diff < 1e-8 then Some sigma
            else
              let (_, up_price), _ = black_scholes 
                ~s:spot_vix ~k:strike ~r:0.0 ~t:maturity ~sigma:(sigma *. 1.01) in
              let vega = (up_price -. bs_price) /. (0.01 *. sigma) in
              if abs_float vega < 1e-10 then None
              else
                let new_sigma = sigma -. diff /. vega in
                if new_sigma <= 0.0 then None
                else newton_vol new_sigma (iter + 1)
        in
        newton_vol 0.5 0
      in
      
      mean_payoff,
      { price = mean_payoff;
        error;
        greeks;
        implied_vol }
  end

  (* VIX futures pricing *)
  module Futures = struct
    let price params state maturity config =
      let paths = StochasticIntegration.Milstein.simulate params state {
        use_antithetic = config.variance_reduction;
        correction_term = true;
        n_paths = config.n_paths;
        dt = config.dt
      } (int_of_float (maturity /. config.dt)) in
      
      let terminal_vixs = List.map (fun path ->
        calc_vix params (List.hd (List.rev path)) config.dt
      ) paths in
      
      let mean_vix = List.fold_left (+.) 0.0 terminal_vixs /. 
                    float_of_int (List.length terminal_vixs) in
      
      let error =
        if config.error_est then
          let variance = List.fold_left (fun acc x ->
            acc +. (x -. mean_vix) ** 2.0
          ) 0.0 terminal_vixs in
          let std_dev = sqrt (variance /. float_of_int (List.length terminal_vixs - 1)) in
          Some (1.96 *. std_dev /. sqrt (float_of_int config.n_paths))
        else None in
      
      mean_vix, error
  end
end

(* Capped volatility process *)
module CappedVolatility = struct
  type cap_parameters = {
    a: float;  (* volatility cap *)
    b: float;  (* drift cap *)
    base_params: parameters;
  }

  (* Compute capped σ_V *)
  let capped_sigma_v params v =
    let open Tensor in
    let base_sigma = Sabr.sigma_v params.base_params v in
    min base_sigma (float_vec params.a)

  (* Compute capped μ_V *)
  let capped_mu_v params v =
    let open Tensor in
    let base_mu = Sabr.mu_v params.base_params v in
    let pos_capped = min base_mu (float_vec params.b) in
    let neg_capped = max base_mu (float_vec (-.params.b)) in
    where_ (base_mu > float_vec 0.0) pos_capped neg_capped

  (* Compute v_hat *)
  let compute_v_hat params =
    let open Float in
    let rho_omega = params.base_params.rho *. params.base_params.omega in
    let term1 = rho_omega +. 
                sqrt (params.a ** 2.0 +. 
                     (params.base_params.rho ** 2.0 -. 1.0) *. 
                     params.base_params.omega ** 2.0) in
    term1 /. (1.0 -. params.base_params.beta)

  (* Evolution step with capped process *)
  let step params state dt =
    let open Tensor in
    let v = state.v in
    let s = state.s in
    
    (* Generate Brownian increments *)
    let dw = randn_like v in
    let drift = capped_mu_v params v * float_vec dt in
    let diffusion = capped_sigma_v params v * dw * float_vec (sqrt dt) in
    
    (* Update volatility *)
    let v_new = v + drift + diffusion in
    
    (* Update asset price with capped volatility *)
    let dz = randn_like s in
    let corr_dz = float_vec params.base_params.rho * dw + 
                  float_vec (sqrt (1.0 -. params.base_params.rho ** 2.0)) * dz in
    
    let s_new = s * exp(
      float_vec params.base_params.beta * v * corr_dz * float_vec (sqrt dt) -
      float_vec 0.5 * (v * v) * float_vec dt
    ) in
    
    { state with v = v_new; s = s_new; t = state.t +. dt }

  (* VIX calculation under capped volatility *)
  let calc_vix params state tau =
    let open Tensor in
    let v = state.v in
    let s = state.s in
    let v_effective = s ** (float_vec (params.base_params.beta -. 1.0)) * v in
    let v_hat = compute_v_hat params in
    let capped_v = min v_effective (float_vec v_hat) in
    sqrt (capped_v * capped_v) |> mean |> to_float0_exn

  (* Option pricing under capped volatility *)
  module Options = struct
    type pricing_result = {
      price: float;
      error: float option;
      greeks: Vix.Options.greeks option;
      implied_vol: float option;
    }

    (* Price VIX call option under capped volatility *)
    let price_call params state ~strike ~maturity ~config =
      let n_steps = int_of_float (maturity /. config.dt) in
      let paths = List.init config.n_paths (fun _ ->
        let rec simulate state acc steps =
          if steps >= n_steps then List.rev (state :: acc)
          else
            let next_state = step params state config.dt in
            simulate next_state (state :: acc) (steps + 1)
        in
        simulate state [] 0
      ) in
      
      let payoffs = List.map (fun path ->
        let terminal_vix = calc_vix params (List.hd (List.rev path)) config.dt in
        max (terminal_vix -. strike) 0.0
      ) paths in
      
      let mean_payoff = List.fold_left (+.) 0.0 payoffs /. 
                       float_of_int config.n_paths in
      
      let error = 
        if config.error_est then
          let variance = List.fold_left (fun acc x ->
            acc +. (x -. mean_payoff) ** 2.0
          ) 0.0 payoffs in
          let std_dev = sqrt (variance /. float_of_int (config.n_paths - 1)) in
          Some (1.96 *. std_dev /. sqrt (float_of_int config.n_paths))
        else None in
      
      (* Calculate Greeks *)
      let greeks = 
        if config.variance_reduction then
          let delta_bump = state.v *. 0.01 in
          let up_state = { state with v = Tensor.(state.v + float_vec delta_bump) } in
          let down_state = { state with v = Tensor.(state.v - float_vec delta_bump) } in
          
          let (up_price, _) = price_call params up_state ~strike ~maturity 
            ~config:{config with error_est=false} in
          let (down_price, _) = price_call params down_state ~strike ~maturity 
            ~config:{config with error_est=false} in
          
          let delta = (up_price -. down_price) /. (2.0 *. delta_bump) in
          let gamma = (up_price -. 2.0 *. mean_payoff +. down_price) /. 
                     (delta_bump *. delta_bump) in
          
          let omega_bump = params.a *. 0.01 in
          let up_params = {params with a = params.a +. omega_bump} in
          let (vega_price, _) = price_call up_params state ~strike ~maturity 
            ~config:{config with error_est=false} in
          let vega = (vega_price -. mean_payoff) /. omega_bump in
          
          let dt = config.dt /. 2.0 in
          let (future_price, _) = price_call params state ~strike 
            ~maturity:(maturity +. dt) ~config:{config with error_est=false} in
          let theta = (future_price -. mean_payoff) /. dt in
          
          Some { Vix.Options.delta; gamma; vega; theta; rho = 0.0 }
        else None in
      
      (* Compute implied volatility *)
      let implied_vol =
        let spot_vix = calc_vix params state config.dt in
        let rec newton_vol sigma iter =
          if iter > 50 then None
          else
            let (bs_price, _), _ = Vix.Options.black_scholes 
              ~s:spot_vix ~k:strike ~r:0.0 ~t:maturity ~sigma in
            let diff = bs_price -. mean_payoff in
            if abs_float diff < 1e-8 then Some sigma
            else
              let ((up_price, _), _) = Vix.Options.black_scholes 
                ~s:spot_vix ~k:strike ~r:0.0 ~t:maturity ~sigma:(sigma *. 1.01) in
              let vega = (up_price -. bs_price) /. (0.01 *. sigma) in
              if abs_float vega < 1e-10 then None
              else
                let new_sigma = sigma -. diff /. vega in
                if new_sigma <= 0.0 then None
                else newton_vol new_sigma (iter + 1)
        in
        newton_vol 0.5 0
      in
      
      mean_payoff,
      { price = mean_payoff;
        error;
        greeks;
        implied_vol }

    (* Price VIX put option under capped volatility *)
    let price_put params state ~strike ~maturity ~config =
      let n_steps = int_of_float (maturity /. config.dt) in
      let paths = List.init config.n_paths (fun _ ->
        let rec simulate state acc steps =
          if steps >= n_steps then List.rev (state :: acc)
          else
            let next_state = step params state config.dt in
            simulate next_state (state :: acc) (steps + 1)
        in
        simulate state [] 0
      ) in
      
      let payoffs = List.map (fun path ->
        let terminal_vix = calc_vix params (List.hd (List.rev path)) config.dt in
        max (strike -. terminal_vix) 0.0
      ) paths in
      
      let mean_payoff = List.fold_left (+.) 0.0 payoffs /. 
                       float_of_int config.n_paths in
      
      (* Error estimation and Greeks calculation similar to calls *)
      let error = 
        if config.error_est then
          let variance = List.fold_left (fun acc x ->
            acc +. (x -. mean_payoff) ** 2.0
          ) 0.0 payoffs in
          let std_dev = sqrt (variance /. float_of_int (config.n_paths - 1)) in
          Some (1.96 *. std_dev /. sqrt (float_of_int config.n_paths))
        else None in
      
      (* Calculate Greeks *)
      let greeks = 
        if config.variance_reduction then
          let delta_bump = state.v *. 0.01 in
          let up_state = { state with v = Tensor.(state.v + float_vec delta_bump) } in
          let down_state = { state with v = Tensor.(state.v - float_vec delta_bump) } in
          
          let (up_price, _) = price_put params up_state ~strike ~maturity 
            ~config:{config with error_est=false} in
          let (down_price, _) = price_put params down_state ~strike ~maturity 
            ~config:{config with error_est=false} in
          
          let delta = (up_price -. down_price) /. (2.0 *. delta_bump) in
          let gamma = (up_price -. 2.0 *. mean_payoff +. down_price) /. 
                     (delta_bump *. delta_bump) in
          
          let omega_bump = params.a *. 0.01 in
          let up_params = {params with a = params.a +. omega_bump} in
          let (vega_price, _) = price_put up_params state ~strike ~maturity 
            ~config:{config with error_est=false} in
          let vega = (vega_price -. mean_payoff) /. omega_bump in
          
          let dt = config.dt /. 2.0 in
          let (future_price, _) = price_put params state ~strike 
            ~maturity:(maturity +. dt) ~config:{config with error_est=false} in
          let theta = (future_price -. mean_payoff) /. dt in
          
          Some { Vix.Options.delta; gamma; vega; theta; rho = 0.0 }
        else None in
      
      (* Compute implied volatility *)
      let implied_vol =
        let spot_vix = calc_vix params state config.dt in
        let rec newton_vol sigma iter =
          if iter > 50 then None
          else
            let (_, bs_price), _ = Vix.Options.black_scholes 
              ~s:spot_vix ~k:strike ~r:0.0 ~t:maturity ~sigma in
            let diff = bs_price -. mean_payoff in
            if abs_float diff < 1e-8 then Some sigma
            else
              let (_, up_price), _ = Vix.Options.black_scholes 
                ~s:spot_vix ~k:strike ~r:0.0 ~t:maturity ~sigma:(sigma *. 1.01) in
              let vega = (up_price -. bs_price) /. (0.01 *. sigma) in
              if abs_float vega < 1e-10 then None
              else
                let new_sigma = sigma -. diff /. vega in
                if new_sigma <= 0.0 then None
                else newton_vol new_sigma (iter + 1)
        in
        newton_vol 0.5 0
      in
      
      mean_payoff,
      { price = mean_payoff;
        error;
        greeks;
        implied_vol }
  end
end

(* Short maturity asymptotics *)
module ShortMaturity = struct
  (* Rate function J_V(K) *)
  let compute_rate_function params v0 k =
    let open Float in
    let v_hat = CappedVolatility.compute_v_hat {
      a = 2.0; b = 1.0; base_params = params 
    } in
    
    if k > v0 then begin
      (* OTM call *)
      if k <= v_hat then
        let rec integrate_sigma_v a b n acc =
          if n <= 0 then acc
          else
            let dx = (b -. a) /. float_of_int n in
            let x = a +. dx *. float_of_int (n - 1) in
            let sigma = Tensor.(
              to_float0_exn (Sabr.sigma_v params (float_vec x))
            ) in
            let term = dx /. (x *. sigma) in
            integrate_sigma_v a b (n - 1) (acc +. term)
        in
        let integral = integrate_sigma_v v0 k 1000 0.0 in
        0.5 *. integral ** 2.0
      else
        0.5 *. (log (k /. v0) /. 2.0) ** 2.0
    end else begin
      (* OTM put *)
      if k >= v_hat then
        let rec integrate_sigma_v a b n acc =
          if n <= 0 then acc
          else
            let dx = (b -. a) /. float_of_int n in
            let x = a +. dx *. float_of_int (n - 1) in
            let sigma = Tensor.(
              to_float0_exn (Sabr.sigma_v params (float_vec x))
            ) in
            let term = dx /. (x *. sigma) in
            integrate_sigma_v a b (n - 1) (acc +. term)
        in
        let integral = integrate_sigma_v k v0 1000 0.0 in
        0.5 *. integral ** 2.0
      else
        0.5 *. (log (k /. v0) /. 2.0) ** 2.0
    end

  (* Short maturity option prices *)
  let asymptotic_prices params v0 k t =
    let j_v = compute_rate_function params v0 k in
    exp (-. j_v /. t)  (* Leading order approximation *)

  (* Implied volatility expansion coefficients *)
  module ImpliedVol = struct
    (* ATM level σ_VIX(0) *)
    let atm_level params v0 =
      Tensor.(
        let s0 = float_vec 1.0 in
        let beta_m1 = params.beta -. 1.0 in
        to_float0_exn (s0 ** (float_vec beta_m1) * (float_vec v0))
      )

    (* VIX skew s_VIX *)
    let skew params v0 =
      let rho_omega = params.rho *. params.omega in
      let beta_m1 = params.beta -. 1.0 in
      v0 *. beta_m1 *. (rho_omega +. beta_m1 *. v0) /.
      (2.0 *. Tensor.(to_float0_exn (Sabr.sigma_v params (float_vec v0))))

    (* VIX convexity κ_VIX *)
    let convexity params v0 =
      let sigma_v0 = Tensor.(to_float0_exn (Sabr.sigma_v params (float_vec v0))) in
      let beta_m1 = params.beta -. 1.0 in
      let term1 = 2.0 *. params.omega ** 3.0 *. params.rho in
      let term2 = beta_m1 *. params.omega ** 2.0 *. 
                 (4.0 +. params.rho ** 2.0) *. v0 in
      let term3 = 4.0 *. beta_m1 ** 2.0 *. params.omega *. 
                 params.rho *. v0 ** 2.0 in
      let term4 = beta_m1 ** 3.0 *. v0 ** 3.0 in
      
      v0 *. beta_m1 /. (3.0 *. sigma_v0 ** 4.0) *. 
      (term1 +. term2 +. term3 +. term4)
  end

  (* Complete implied volatility expansion *)
  let implied_vol_expansion params v0 k t =
    let x = log (k /. v0) in
    let sigma0 = ImpliedVol.atm_level params v0 in
    let s = ImpliedVol.skew params v0 in
    let kappa = ImpliedVol.convexity params v0 in
    sigma0 +. s *. x +. 0.5 *. kappa *. x *. x
end

(* Model calibration *)
module Calibration = struct
  type market_data = {
    strikes: float array;
    maturities: float array;
    call_prices: float array array;  (* [maturity_idx][strike_idx] *)
    put_prices: float array array;
    call_ivols: float array array option;
    put_ivols: float array array option;
  }

  type calibration_config = {
    max_iter: int;
    tolerance: float;
    regularization: float;
    method_name: string;
  }

  type calibration_result = {
    parameters: parameters;
    error: float;
    n_iterations: int;
    convergence: bool;
    fit_quality: fit_metrics;
  }
  and fit_metrics = {
    rmse: float;
    max_error: float;
    avg_error: float;
    r_squared: float;
  }

  (* Loss function calculation *)
  let calculate_loss params market_data config =
    let open Float in
    let total_loss = ref 0.0 in
    let n_points = ref 0 in
    
    (* Loop over maturities and strikes *)
    Array.iteri (fun i maturity ->
      Array.iteri (fun j strike ->
        let state = {
          s = Tensor.float_vec 1.0;
          v = Tensor.float_vec params.omega;
          t = 0.0
        } in
        
        (* Price options *)
        let model_call, _ = Vix.Options.price_call params state 
          ~strike ~maturity 
          ~config:{
            n_paths=1000; 
            dt=0.01; 
            variance_reduction=true;
            error_est=false
          } in
        
        let model_put, _ = Vix.Options.price_put params state
          ~strike ~maturity
          ~config:{
            n_paths=1000;
            dt=0.01;
            variance_reduction=true;
            error_est=false
          } in
        
        (* Compute squared errors *)
        let call_error = (model_call -. market_data.call_prices.(i).(j)) ** 2.0 in
        let put_error = (model_put -. market_data.put_prices.(i).(j)) ** 2.0 in
        
        (* Add regularization *)
        let reg_term = config.regularization *. (
          params.omega ** 2.0 +. 
          params.beta ** 2.0 +. 
          params.rho ** 2.0
        ) in
        
        total_loss := !total_loss +. call_error +. put_error +. reg_term;
        incr n_points
      ) market_data.strikes
    ) market_data.maturities;
    
    !total_loss /. float_of_int !n_points

  (* Parameter constraints *)
  let enforce_constraints params =
    let open Float in
    {
      beta = max 0.0 (min 1.0 params.beta);
      omega = max 0.0 params.omega;
      rho = max (-1.0) (min 1.0 params.rho);
      alpha = params.alpha;
      delta = params.delta;
    }

  (* Levenberg-Marquardt optimization *)
  let optimize initial_params market_data config =
    let open Float in
    let lambda = ref 0.1 in  (* Damping parameter *)
    
    let rec optimize_step params iter best_loss =
      if iter >= config.max_iter then
        Error "Maximum iterations reached"
      else
        (* Compute gradients using finite differences *)
        let h = 1e-6 in
        let params_beta = { params with beta = params.beta +. h } in
        let params_omega = { params with omega = params.omega +. h } in
        let params_rho = { params with rho = params.rho +. h } in
        
        let base_loss = calculate_loss params market_data config in
        let d_beta = (calculate_loss params_beta market_data config -. base_loss) /. h in
        let d_omega = (calculate_loss params_omega market_data config -. base_loss) /. h in
        let d_rho = (calculate_loss params_rho market_data config -. base_loss) /. h in
        
        (* Compute approximate Hessian *)
        let h = sqrt h in
        let hessian = Array.make_matrix 3 3 0.0 in
        for i = 0 to 2 do
          for j = 0 to 2 do
            let params_i = match i with
              | 0 -> { params with beta = params.beta +. h }
              | 1 -> { params with omega = params.omega +. h }
              | _ -> { params with rho = params.rho +. h }
            in
            let params_j = match j with
              | 0 -> { params with beta = params.beta +. h }
              | 1 -> { params with omega = params.omega +. h }
              | _ -> { params with rho = params.rho +. h }
            in
            let params_ij = match (i, j) with
              | (0, 0) -> { params with beta = params.beta +. 2.0 *. h }
              | (1, 1) -> { params with omega = params.omega +. 2.0 *. h }
              | (2, 2) -> { params with rho = params.rho +. 2.0 *. h }
              | (0, 1) | (1, 0) -> 
                  { params with beta = params.beta +. h; omega = params.omega +. h }
              | (0, 2) | (2, 0) -> 
                  { params with beta = params.beta +. h; rho = params.rho +. h }
              | (1, 2) | (2, 1) -> 
                  { params with omega = params.omega +. h; rho = params.rho +. h }
              | _ -> params
            in
            
            let f_ij = calculate_loss params_ij market_data config in
            let f_i = calculate_loss params_i market_data config in
            let f_j = calculate_loss params_j market_data config in
            
            hessian.(i).(j) <- (f_ij -. f_i -. f_j +. base_loss) /. (h *. h)
          done
        done;
        
        (* Add damping *)
        for i = 0 to 2 do
          hessian.(i).(i) <- hessian.(i).(i) +. !lambda
        done;
        
        (* Solve system for update *)
        let gradient = [|d_beta; d_omega; d_rho|] in
        let solve_system a b =
          let n = Array.length a in
          let x = Array.make n 0.0 in
          
          (* Gaussian elimination with pivoting *)
          for i = 0 to n-1 do
            (* Find pivot *)
            let max_idx = ref i in
            for j = i+1 to n-1 do
              if abs_float a.(j).(i) > abs_float a.(!max_idx).(i) then
                max_idx := j
            done;
            
            (* Swap rows *)
            if !max_idx <> i then begin
              let temp_row = a.(i) in
              a.(i) <- a.(!max_idx);
              a.(!max_idx) <- temp_row;
              let temp_b = b.(i) in
              b.(i) <- b.(!max_idx);
              b.(!max_idx) <- temp_b
            end;
            
            (* Eliminate column *)
            for j = i+1 to n-1 do
              let factor = a.(j).(i) /. a.(i).(i) in
              b.(j) <- b.(j) -. factor *. b.(i);
              for k = i to n-1 do
                a.(j).(k) <- a.(j).(k) -. factor *. a.(i).(k)
              done
            done
          done;
          
          (* Back substitution *)
          for i = n-1 downto 0 do
            let sum = ref 0.0 in
            for j = i+1 to n-1 do
              sum := !sum +. a.(i).(j) *. x.(j)
            done;
            x.(i) <- (b.(i) -. !sum) /. a.(i).(i)
          done;
          x
        in
        
        let update = solve_system hessian gradient in
        
        (* Update parameters with line search *)
        let rec line_search alpha n =
          if n > 10 then alpha
          else
            let new_params = {
              params with
              beta = params.beta -. alpha *. update.(0);
              omega = params.omega -. alpha *. update.(1);
              rho = params.rho -. alpha *. update.(2);
            } |> enforce_constraints in
            
            let new_loss = calculate_loss new_params market_data config in
            if new_loss < base_loss then alpha
            else line_search (alpha /. 2.0) (n + 1)
        in
        
        let alpha = line_search 1.0 0 in
        let new_params = {
          params with
          beta = params.beta -. alpha *. update.(0);
          omega = params.omega -. alpha *. update.(1);
          rho = params.rho -. alpha *. update.(2);
        } |> enforce_constraints in
        
        let new_loss = calculate_loss new_params market_data config in
        
        if abs_float (new_loss -. best_loss) < config.tolerance then
          Ok (new_params, new_loss, iter)
        else begin
          (* Update damping parameter *)
          lambda := if new_loss < best_loss then !lambda /. 10.0
        (* Continuation of Calibration module *)
          else !lambda *. 10.0;
          optimize_step new_params (iter + 1) (min best_loss new_loss)
        end
    in
    
    match optimize_step initial_params 0 Float.infinity with
    | Error msg -> Error msg
    | Ok (params, loss, iters) ->
        (* Compute fit metrics *)
        let compute_fit_metrics params market_data =
          let open Float in
          let n_total = ref 0 in
          let sum_squared_error = ref 0.0 in
          let max_error = ref 0.0 in
          let sum_error = ref 0.0 in
          let sum_squared_total = ref 0.0 in
          let mean_market = ref 0.0 in
          
          (* First pass to compute mean *)
          Array.iteri (fun i maturity ->
            Array.iteri (fun j strike ->
              let market_call = market_data.call_prices.(i).(j) in
              let market_put = market_data.put_prices.(i).(j) in
              mean_market := !mean_market +. market_call +. market_put;
              incr n_total;
              incr n_total
            ) market_data.strikes
          ) market_data.maturities;
          
          mean_market := !mean_market /. float_of_int !n_total;
          
          (* Second pass to compute metrics *)
          Array.iteri (fun i maturity ->
            Array.iteri (fun j strike ->
              let state = {
                s = Tensor.float_vec 1.0;
                v = Tensor.float_vec params.omega;
                t = 0.0
              } in
              
              let model_call, _ = Vix.Options.price_call params state 
                ~strike ~maturity 
                ~config:{n_paths=1000; dt=0.01; 
                        variance_reduction=true; error_est=false} in
              
              let model_put, _ = Vix.Options.price_put params state
                ~strike ~maturity
                ~config:{n_paths=1000; dt=0.01; 
                        variance_reduction=true; error_est=false} in
              
              let market_call = market_data.call_prices.(i).(j) in
              let market_put = market_data.put_prices.(i).(j) in
              
              let error_call = abs_float (model_call -. market_call) in
              let error_put = abs_float (model_put -. market_put) in
              
              max_error := max !max_error (max error_call error_put);
              sum_error := !sum_error +. error_call +. error_put;
              sum_squared_error := !sum_squared_error +. error_call ** 2.0 +. 
                                 error_put ** 2.0;
              sum_squared_total := !sum_squared_total +. 
                                 (market_call -. !mean_market) ** 2.0 +.
                                 (market_put -. !mean_market) ** 2.0
            ) market_data.strikes
          ) market_data.maturities;
          
          let rmse = sqrt (!sum_squared_error /. float_of_int !n_total) in
          let avg_error = !sum_error /. float_of_int !n_total in
          let r_squared = 1.0 -. !sum_squared_error /. !sum_squared_total in
          
          { rmse; max_error = !max_error; avg_error; r_squared }
        in
        
        let fit_metrics = compute_fit_metrics params market_data in
        Ok {
          parameters = params;
          error = loss;
          n_iterations = iters;
          convergence = iters < config.max_iter;
          fit_quality = fit_metrics
        }

  module Analysis = struct
    (* Compute model implied volatility surface *)
    let compute_vol_surface params ~strikes ~maturities =
      let n_strikes = Array.length strikes in
      let n_maturities = Array.length maturities in
      let surface = Array.make_matrix n_maturities n_strikes 0.0 in
      
      let state = {
        s = Tensor.float_vec 1.0;
        v = Tensor.float_vec params.omega;
        t = 0.0
      } in
      
      for i = 0 to n_maturities - 1 do
        for j = 0 to n_strikes - 1 do
          let strike = strikes.(j) in
          let maturity = maturities.(i) in
          
          let _, result = Vix.Options.price_call params state 
            ~strike ~maturity
            ~config:{n_paths=1000; dt=0.01; 
                    variance_reduction=true; error_est=false} in
          
          match result.implied_vol with
          | Some iv -> surface.(i).(j) <- iv
          | None -> surface.(i).(j) <- Float.nan
        done
      done;
      surface

    (* Analyze calibration stability *)
    let analyze_stability params market_data ~n_trials =
      let results = Array.init n_trials (fun _ ->
        (* Perturb initial parameters *)
        let perturbed = {
          params with
          beta = params.beta *. (1.0 +. 0.1 *. (Random.float 2.0 -. 1.0));
          omega = params.omega *. (1.0 +. 0.1 *. (Random.float 2.0 -. 1.0));
          rho = params.rho *. (1.0 +. 0.1 *. (Random.float 2.0 -. 1.0));
        } |> enforce_constraints in
        
        match optimize perturbed market_data 
                {max_iter=100; tolerance=1e-6; 
                 regularization=1e-4; method_name="LM"} with
        | Ok result -> Some result
        | Error _ -> None
      ) in
      
      let successful = Array.fold_left (fun acc result ->
        match result with
        | Some _ -> acc + 1
        | None -> acc
      ) 0 results in
      
      let param_stats = Array.fold_left (fun (beta_acc, omega_acc, rho_acc) result ->
        match result with
        | Some r -> 
            (beta_acc @ [r.parameters.beta],
             omega_acc @ [r.parameters.omega],
             rho_acc @ [r.parameters.rho])
        | None -> (beta_acc, omega_acc, rho_acc)
      ) ([], [], []) results in
      
      let compute_stats values =
        let n = float_of_int (List.length values) in
        let mean = List.fold_left (+.) 0.0 values /. n in
        let std = sqrt (List.fold_left (fun acc x ->
          acc +. (x -. mean) ** 2.0
        ) 0.0 values /. n) in
        mean, std
      in
      
      let (beta_mean, beta_std) = compute_stats (let (b,_,_) = param_stats in b) in
      let (omega_mean, omega_std) = compute_stats (let (_,o,_) = param_stats in o) in
      let (rho_mean, rho_std) = compute_stats (let (_,_,r) = param_stats in r) in
      
      {
        success_rate = float_of_int successful /. float_of_int n_trials;
        parameter_stability = {
          beta = (beta_mean, beta_std);
          omega = (omega_mean, omega_std);
          rho = (rho_mean, rho_std)
        }
      }

    (* Analyze model fit across strikes and maturities *)
    let analyze_fit_quality params market_data =
      let strikes = market_data.strikes in
      let maturities = market_data.maturities in
      
      (* Compute errors across strike slices *)
      let strike_errors = Array.map (fun strike ->
        let errors = ref [] in
        Array.iteri (fun i maturity ->
          let state = {
            s = Tensor.float_vec 1.0;
            v = Tensor.float_vec params.omega;
            t = 0.0
          } in
          
          let model_call, _ = Vix.Options.price_call params state 
            ~strike ~maturity
            ~config:{n_paths=1000; dt=0.01; 
                    variance_reduction=true; error_est=false} in
          
          let model_put, _ = Vix.Options.price_put params state
            ~strike ~maturity
            ~config:{n_paths=1000; dt=0.01; 
                    variance_reduction=true; error_est=false} in
          
          let market_call = market_data.call_prices.(i).(Array.index strike strikes) in
          let market_put = market_data.put_prices.(i).(Array.index strike strikes) in
          
          errors := (abs_float (model_call -. market_call)) :: !errors;
          errors := (abs_float (model_put -. market_put)) :: !errors
        ) maturities;
        let mean_error = List.fold_left (+.) 0.0 !errors /. 
                        float_of_int (List.length !errors) in
        strike, mean_error
      ) strikes in
      
      (* Compute errors across maturity slices *)
      let maturity_errors = Array.map (fun maturity ->
        let errors = ref [] in
        Array.iteri (fun j strike ->
          let state = {
            s = Tensor.float_vec 1.0;
            v = Tensor.float_vec params.omega;
            t = 0.0
          } in
          
          let model_call, _ = Vix.Options.price_call params state 
            ~strike ~maturity
            ~config:{n_paths=1000; dt=0.01; 
                    variance_reduction=true; error_est=false} in
          
          let model_put, _ = Vix.Options.price_put params state
            ~strike ~maturity
            ~config:{n_paths=1000; dt=0.01; 
                    variance_reduction=true; error_est=false} in
          
          let i = Array.index maturity maturities in
          let market_call = market_data.call_prices.(i).(j) in
          let market_put = market_data.put_prices.(i).(j) in
          
          errors := (abs_float (model_call -. market_call)) :: !errors;
          errors := (abs_float (model_put -. market_put)) :: !errors
        ) strikes;
        let mean_error = List.fold_left (+.) 0.0 !errors /. 
                        float_of_int (List.length !errors) in
        maturity, mean_error
      ) maturities in
      
      {
        strike_slice_errors = strike_errors;
        maturity_slice_errors = maturity_errors;
        total_rmse = compute_fit_metrics params market_data
      }
  end
end