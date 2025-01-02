open Torch

type model_params = {
  v0: float;
  vbar: float;
  kappa: float;
  xi: float;
  rho: float;
}

type grid_params = {
  s_min: float;
  s_max: float;
  v_min: float;
  v_max: float;
  ns: int;
  nv: int;
  nt: int;
  dt: float;
}

type solver_type = [ `ADI_HV | `ADI_MCS | `BDF3 ]

type option_data = {
  strike: float;
  expiry: float;
  market_price: float;
  option_type: [ `Call | `Put ];
}

module GridGen = struct
  let calc_bounds params data =
    let s_max_factor = 5.0 in
    let v_max_factor = 4.0 in
    let sigma_est = 0.5 *. (sqrt params.v0 +. sqrt params.vbar) in
    let s_max = data.strike *. exp(s_max_factor *. sigma_est) in
    let s_min = 0.0 in
    let v_min = 0.0 in
    let v_max = params.vbar *. (1.0 +. v_max_factor *. sqrt(2.0 *. params.kappa *. params.vbar /. 
                                                           (params.xi *. params.xi))) in
    s_min, s_max, v_min, v_max

  let gen_s_grid params ~strike =
    let bs = 4.5 in (* Non-uniformity parameter *)
    let points = Tensor.zeros [|params.ns|] in
    for i = 0 to params.ns - 1 do
      let alpha = float_of_int i /. float_of_int (params.ns - 1) in
      let s = params.s_min +. strike *. 
        (1.0 +. sinh(bs *. (alpha -. 0.5)) /. sinh(bs *. 0.5)) in
      Tensor.set points [|i|] s
    done;
    points

  let gen_v_grid params ~v0 =
    let bv = 8.5 in (* High non-uniformity for v-direction *)
    let points = Tensor.zeros [|params.nv|] in
    for j = 0 to params.nv - 1 do
      let alpha = float_of_int j /. float_of_int (params.nv - 1) in
      let v = v0 *. (1.0 +. sinh(bv *. alpha) /. sinh bv) in
      Tensor.set points [|j|] v
    done;
    points
end

module FiniteDiff = struct
  let first_deriv_coeff h_i h_ip1 =
    let c_minus = -.h_ip1 /. (h_i *. (h_i +. h_ip1)) in
    let c_center = (h_ip1 -. h_i) /. (h_i *. h_ip1) in
    let c_plus = h_i /. (h_ip1 *. (h_i +. h_ip1)) in
    c_minus, c_center, c_plus

  let second_deriv_coeff h_i h_ip1 =
    let c_minus = 2.0 /. (h_i *. (h_i +. h_ip1)) in
    let c_center = -.2.0 /. (h_i *. h_ip1) in
    let c_plus = 2.0 /. (h_ip1 *. (h_i +. h_ip1)) in
    c_minus, c_center, c_plus

  let mixed_deriv_coeff h_s h_sp1 h_v h_vp1 =
    let d = h_sp1 *. h_v +. h_s *. h_vp1 in
    1.0 /. d
end

module ADI = struct
  type operator_split = {
    a0: Tensor.t;
    a1: Tensor.t;
    a2: Tensor.t;
    b: Tensor.t;
  }

  module HV = struct
    let step ~dt matrices v_prev =
      let theta = 1.0 -. sqrt 2.0 /. 2.0 in

      let y0 = Tensor.(v_prev + dt * (matmul matrices.a0 v_prev + 
                                     matmul matrices.a1 v_prev + 
                                     matmul matrices.a2 v_prev + 
                                     matrices.b)) in
      
      let i_minus_theta_dt_a1 = Tensor.(eye_like matrices.a1 - scalar_float (theta *. dt) * matrices.a1) in
      let i_minus_theta_dt_a2 = Tensor.(eye_like matrices.a2 - scalar_float (theta *. dt) * matrices.a2) in
      
      let y1 = Tensor.solve i_minus_theta_dt_a1 
        Tensor.(y0 + scalar_float (theta *. dt) * matmul matrices.a1 v_prev) in
      
      let y2 = Tensor.solve i_minus_theta_dt_a2
        Tensor.(y1 + scalar_float (theta *. dt) * matmul matrices.a2 v_prev) in
      
      let y_tilde = Tensor.(y0 + scalar_float 0.5 * dt * 
        (matmul matrices.a0 (y2 - v_prev) +
         matmul matrices.a1 (y2 - v_prev) +
         matmul matrices.a2 (y2 - v_prev))) in
         
      let y_tilde1 = Tensor.solve i_minus_theta_dt_a1
        Tensor.(y_tilde + scalar_float (theta *. dt) * matmul matrices.a1 y2) in
      
      let y_tilde2 = Tensor.solve i_minus_theta_dt_a2
        Tensor.(y_tilde1 + scalar_float (theta *. dt) * matmul matrices.a2 y2) in

      y_tilde2
  end

  module MCS = struct
    let step ~dt matrices v_prev =
      let theta = 1.0 /. 3.0 in
      
      let y0 = Tensor.(v_prev + dt * (matmul matrices.a0 v_prev + 
                                     matmul matrices.a1 v_prev + 
                                     matmul matrices.a2 v_prev + 
                                     matrices.b)) in
      
      let i_minus_theta_dt_a1 = Tensor.(eye_like matrices.a1 - scalar_float (theta *. dt) * matrices.a1) in
      let i_minus_theta_dt_a2 = Tensor.(eye_like matrices.a2 - scalar_float (theta *. dt) * matrices.a2) in
      
      let y1 = Tensor.solve i_minus_theta_dt_a1
        Tensor.(y0 + scalar_float (theta *. dt) * matmul matrices.a1 v_prev) in
        
      let y2 = Tensor.solve i_minus_theta_dt_a2
        Tensor.(y1 + scalar_float (theta *. dt) * matmul matrices.a2 v_prev) in
      
      let y_hat = Tensor.(y0 + scalar_float theta * dt * 
        matmul matrices.a0 (y2 - v_prev)) in
        
      let y_tilde = Tensor.(y_hat + scalar_float (0.5 -. theta) * dt * 
        (matmul matrices.a0 (y2 - v_prev) +
         matmul matrices.a1 (y2 - v_prev) +
         matmul matrices.a2 (y2 - v_prev))) in
         
      let y_tilde1 = Tensor.solve i_minus_theta_dt_a1
        Tensor.(y_tilde + scalar_float (theta *. dt) * matmul matrices.a1 y2) in
        
      let y_tilde2 = Tensor.solve i_minus_theta_dt_a2
        Tensor.(y_tilde1 + scalar_float (theta *. dt) * matmul matrices.a2 y2) in
      
      y_tilde2
  end
end

module BDF3 = struct
  let implicit_euler_step ~dt matrix v_prev rhs =
    let i_minus_dt_a = Tensor.(eye_like matrix - scalar_float dt * matrix) in
    Tensor.solve i_minus_dt_a Tensor.(v_prev + scalar_float dt * rhs)

  let bdf2_step ~dt matrix v_prev v_prev2 rhs =
    let coeff = 1.5 /. dt in
    let i_minus_dt_a = Tensor.(scalar_float coeff * eye_like matrix - matrix) in
    let rhs_mod = Tensor.(scalar_float coeff * (scalar_float 2.0 * v_prev - 
                         scalar_float 0.5 * v_prev2) + rhs) in
    Tensor.solve i_minus_dt_a rhs_mod

  let solve params matrices initial_values =
    let n_steps = params.grid_params.nt in
    let dt = params.grid_params.dt in
    
    let v_nm2 = ref initial_values in
    let v_nm1 = ref (implicit_euler_step ~dt:dt matrices !v_nm2 Tensor.zeros_like(initial_values)) in
    let v_n = ref (bdf2_step ~dt matrices !v_nm1 !v_nm2 Tensor.zeros_like(initial_values)) in
    
    for _ = 3 to n_steps do
      let coeff = 11.0 /. (6.0 *. dt) in
      let i_minus_dt_a = Tensor.(scalar_float coeff * eye_like matrices.a0 - matrices.a0) in
      let rhs = Tensor.(scalar_float coeff * (scalar_float 3.0 * !v_n - 
                scalar_float 1.5 * !v_nm1 + scalar_float (1.0 /. 3.0) * !v_nm2) + matrices.b) in
      
      let v_np1 = Tensor.solve i_minus_dt_a rhs in
      v_nm2 := !v_nm1;
      v_nm1 := !v_n;
      v_n := v_np1
    done;
    
    !v_n
end

let build_coefficient_matrix params grid_params s_grid v_grid =
  let ns = grid_params.ns in
  let nv = grid_params.nv in
  let n = ns * nv in
  
  let a0 = Tensor.zeros [|n; n|] in
  let a1 = Tensor.zeros [|n; n|] in
  let a2 = Tensor.zeros [|n; n|] in
  let b = Tensor.zeros [|n|] in
  
  for j = 0 to nv-1 do
    let v = Tensor.get v_grid [|j|] in
    for i = 0 to ns-1 do
      let s = Tensor.get s_grid [|i|] in
      let idx = j * ns + i in
      
      if i > 0 && i < ns-1 && j > 0 && j < nv-1 then begin
        (* Calculate steps *)
        let h_s = Tensor.get s_grid [|i+1|] -. s in
        let h_v = Tensor.get v_grid [|j+1|] -. v in
        
        (* Diffusion coefficients *)
        let ds = 0.5 *. s *. s in
        let dv = 0.5 *. params.xi *. params.xi *. v *. v in
        let msv = params.rho *. params.xi *. s *. (v ** 1.5) in
        
        (* Convection coefficients *)
        let cs = (params.rho -. params.kappa) *. s in
        let cv = params.kappa *. (params.vbar -. v) in
        
        (* Get finite difference coefficients *)
        let s_minus, s_center, s_plus = FiniteDiff.second_deriv_coeff h_s h_s in
        let v_minus, v_center, v_plus = FiniteDiff.second_deriv_coeff h_v h_v in
        let m_coeff = FiniteDiff.mixed_deriv_coeff h_s h_s h_v h_v in
        
        (* Fill matrices *)
        (* S direction *)
        Tensor.set_2d a1 [|idx; idx-1|] (ds *. s_minus +. cs *. s_minus);
        Tensor.set_2d a1 [|idx; idx|] (ds *. s_center +. cs *. s_center);
        Tensor.set_2d a1 [|idx; idx+1|] (ds *. s_plus +. cs *. s_plus);
        
        (* V direction *)
        Tensor.set_2d a2 [|idx; idx-ns|] (dv *. v_minus +. cv *. v_minus);
        Tensor.set_2d a2 [|idx; idx|] (dv *. v_center +. cv *. v_center);
        Tensor.set_2d a2 [|idx; idx+ns|] (dv *. v_plus +. cv *. v_plus);
        
        (* Mixed derivatives *)
        Tensor.set_2d a0 [|idx; idx-ns-1|] (-. msv *. m_coeff);
        Tensor.set_2d a0 [|idx; idx+ns+1|] (-. msv *. m_coeff);
        Tensor.set_2d a0 [|idx; idx-ns+1|] (msv *. m_coeff);
        Tensor.set_2d a0 [|idx; idx+ns-1|] (msv *. m_coeff);
      end
    done
  done;
  
  {ADI.a0; a1; a2; b}

module Richardson = struct
  type extrap_config = {
    space_ratio: float;
    time_ratio: float;
    space_order: int;
    time_order: int;
  }

  let extrapolate solver params config =
    (* Solve on coarse grid *)
    let coarse_params = {params with
      grid_params = {
        params.grid_params with
        ns = params.grid_params.ns / int_of_float config.space_ratio;
        nv = params.grid_params.nv / int_of_float config.space_ratio;
        nt = params.grid_params.nt / int_of_float config.time_ratio
      }
    } in
    let coarse_sol = solver coarse_params in
    
    (* Solve on fine grid *)
    let fine_sol = solver params in
    
    (* Calculate extrapolation coefficients *)
    let rs = config.space_ratio ** float config.space_order in
    let rt = config.time_ratio ** float config.time_order in
    let cs = rs /. (rs -. 1.0) in
    let ct = rt /. (rt -. 1.0) in
    
    (* Combine solutions and estimate error *)
    let extrap_sol = Tensor.((scalar_float (cs *. ct) * fine_sol) - 
                            (scalar_float ((cs -. 1.0) *. (ct -. 1.0)) * coarse_sol)) in
    
    let error_est = Tensor.(abs(fine_sol - coarse_sol)) |> 
                    Tensor.mean |> 
                    Tensor.to_float0_exn in
    
    extrap_sol, error_est
end

module Calibration = struct
  type calibration_mode = Price_Based | IV_Based | Hybrid
  
  type calibration_result = {
    fitted_params: model_params;
    rmse: float;
    iterations: int;
    time_taken: float;
    stability_metric: float;
    negative_values: int;
    parameter_path: model_params list;
  }

  (* Black-Scholes implied volatility calculation *)
  let calc_implied_vol price data =
    let open Float in
    let rec newton_iter sigma iter =
      if iter > 50 then sigma
      else
        let d1 = (log(data.strike/.price) +. (0.5*.sigma*.sigma)*.data.expiry) /. 
                 (sigma *. sqrt data.expiry) in
        let d2 = d1 -. sigma *. sqrt data.expiry in
        
        let bs_price = match data.option_type with
        | `Call -> price *. Torch.Tensor.(exp (scalar_float (-0.5 *. d1 *. d1))) -. 
                   data.strike *. exp(-0.5 *. d2 *. d2)
        | `Put -> data.strike *. exp(-0.5 *. d2 *. d2) -. 
                 price *. Torch.Tensor.(exp (scalar_float (-0.5 *. d1 *. d1)))
        in
        
        let vega = price *. sqrt data.expiry *. 
                   Torch.Tensor.(exp (scalar_float (-0.5 *. d1 *. d1))) in
        let diff = bs_price -. data.market_price in
        
        if abs diff < 1e-7 then sigma
        else newton_iter (sigma -. diff /. vega) (iter + 1)
    in
    newton_iter 0.5 0

  let objective mode params data =
    match mode with
    | Price_Based ->
        let sum_sq_errs = List.fold_left (fun acc opt ->
          let price = Pricer.price params opt in
          acc +. ((price -. opt.market_price) ** 2.0)
        ) 0.0 data in
        sqrt (sum_sq_errs /. float_of_int (List.length data))
        
    | IV_Based ->
        let sum_sq_errs = List.fold_left (fun acc opt ->
          let price = Pricer.price params opt in
          let model_iv = calc_implied_vol price opt in
          let market_iv = calc_implied_vol opt.market_price opt in
          acc +. ((model_iv -. market_iv) ** 2.0)
        ) 0.0 data in
        sqrt (sum_sq_errs /. float_of_int (List.length data))
        
    | Hybrid ->
        0.3 *. objective Price_Based params data +.
        0.7 *. objective IV_Based params data

  let calibrate params data mode =
    let start_time = Unix.gettimeofday() in
    
    let param_path = ref [params] in
    let neg_vals = ref 0 in
    let iter = ref 0 in
    
    (* L-BFGS optimization *)
    let rec optimize params step =
      if !iter >= 1000 then Error "Max iterations exceeded"
      else begin
        iter := !iter + 1;
        
        let obj = objective mode params data in
        if obj < 1e-4 then begin
          let end_time = Unix.gettimeofday() in
          Ok {
            fitted_params = params;
            rmse = obj;
            iterations = !iter;
            time_taken = end_time -. start_time;
            stability_metric = 0.0; (* Calculate from param_path *)
            negative_values = !neg_vals;
            parameter_path = List.rev !param_path;
          }
        end else begin
          let new_params = optimize_step params obj step in
          param_path := new_params :: !param_path;
          if check_negative_values new_params data then
            neg_vals := !neg_vals + 1;
          optimize new_params (step *. 0.95)
        end
      end
    in
    
    optimize params 0.1
end

module ModelComparison = struct
  type model = 
    | GARCH of model_params
    | Heston of model_params
    | PModel of {base: model_params; p: float}

  type comparison_metrics = {
    rmse_iv: float;
    avg_calib_time: float;
    param_stability: float;
    short_term_fit: float;
    long_term_fit: float;
    smile_coverage: float;
    feller_ratio: float option;
  }

  (* Calculate all metrics for a model *)
  let analyze_model model data =
    let price_fn = match model with
    | GARCH params -> Pricer.price params
    | Heston params -> Pricer.price_heston params
    | PModel {base; p} -> Pricer.price_p_model base p
    in
    
    let short_term = List.filter (fun d -> d.expiry <= 0.25) data in
    let long_term = List.filter (fun d -> d.expiry > 0.25) data in
    
    let rmse = Calibration.objective IV_Based price_fn data in
    let start_time = Unix.gettimeofday() in
    let _ = Calibration.calibrate price_fn data Calibration.IV_Based in
    let end_time = Unix.gettimeofday() in
    
    {
      rmse_iv = rmse;
      avg_calib_time = end_time -. start_time;
      param_stability = calc_stability model data;
      short_term_fit = Calibration.objective IV_Based price_fn short_term;
      long_term_fit = Calibration.objective IV_Based price_fn long_term;
      smile_coverage = calc_smile_coverage model data;
      feller_ratio = match model with
        | Heston p -> Some (2.0 *. p.kappa *. p.vbar /. (p.xi *. p.xi))
        | _ -> None
    }

  (* Compare multiple models *)
  let compare_models models data =
    List.map (fun model -> 
      model, analyze_model model data
    ) models
end