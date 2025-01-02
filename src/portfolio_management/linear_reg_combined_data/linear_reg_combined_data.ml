open Torch

type dimensions = {
  p_o : int;
  p_c : int;
  q : int;
}

type variable_type = 
  | Outside
  | Common 
  | CommonVar

type dataset = {
  features : Tensor.t;
  target : Tensor.t;
  common_vars : Tensor.t option;
}

type model_params = {
  lambda : float;
  include_common : bool;
}

let compute_residuals x y =
  let beta = Tensor.mm (Tensor.pinverse (Tensor.matmul x (Tensor.transpose x ~dim0:1 ~dim1:0)))
                      (Tensor.matmul x (Tensor.transpose y ~dim0:1 ~dim1:0)) in
  let pred = Tensor.matmul x beta in
  Tensor.sub y pred

let partial_residuals target regressors controls =
  let target_resid = compute_residuals controls target in
  let reg_resid = compute_residuals controls regressors in
  target_resid, reg_resid

let mean tensor = 
  Tensor.mean tensor ~dim:[0] ~keepdim:true

let var tensor =
  let centered = Tensor.sub tensor (mean tensor) in
  Tensor.mean (Tensor.mul centered centered) ~dim:[0] ~keepdim:true

let cov x y =
  let x_centered = Tensor.sub x (mean x) in
  let y_centered = Tensor.sub y (mean y) in
  let n = Float.of_int (Tensor.size x 0) in
  Tensor.mm (Tensor.transpose x_centered ~dim0:1 ~dim1:0) y_centered
  |> Tensor.div_scalar (n -. 1.)

let quantile tensor p =
  let sorted = Tensor.sort tensor ~dim:0 ~descending:false |> fst in
  let n = Tensor.size tensor 0 in
  let idx = Float.to_int (p *. Float.of_int (n - 1)) in
  Tensor.get sorted idx |> Tensor.to_float0_exn

let decompose_variables target regressors common_vars =
  (* Compute conditional expectations *)
  let gamma_target = 
    let beta = Tensor.matmul 
      (Tensor.pinverse common_vars) 
      (Tensor.reshape target ~shape:[-1; 1]) in
    Tensor.matmul common_vars beta |> Tensor.squeeze ~dim:[1]
  in
  
  let gamma_reg = 
    let beta = Tensor.matmul 
      (Tensor.pinverse common_vars) 
      regressors in
    Tensor.matmul common_vars beta
  in
  
  (* Compute residuals *)
  let nu_target = Tensor.sub target gamma_target in
  let nu_reg = Tensor.sub regressors gamma_reg in
  
  ((gamma_target, gamma_reg), (nu_target, nu_reg))

let conditional_quantile_function values conditioning u =
  let n = Tensor.size values 0 in
  let kernel_bandwidth = 
    1.06 *. (Tensor.std conditioning ~unbiased:true |> Tensor.to_float0_exn) 
    *. Float.pow (Float.of_int n) (-0.2)
  in
  
  let kernel x_i x = 
    let diff = Tensor.sub x_i x in
    Tensor.exp (Tensor.div_scalar (Tensor.mul diff diff) 
                                 (-2. *. kernel_bandwidth *. kernel_bandwidth))
  in
  
  let compute_weighted_quantile w_point =
    let sorted_values, indices = Tensor.sort values ~dim:0 ~descending:false in
    let sorted_cond = Tensor.index_select conditioning ~dim:0 ~index:indices in
    let weights = kernel w_point sorted_cond in
    let cumsum = Tensor.cumsum weights ~dim:0 in
    let normalized = Tensor.div cumsum (Tensor.sum weights ~dim:[0]) in
    
    let diffs = Tensor.sub normalized (Tensor.full_like normalized u) 
               |> Tensor.abs in
    let min_idx = Tensor.argmin diffs ~dim:0 |> Tensor.to_int0_exn in
    Tensor.get sorted_values min_idx |> Tensor.to_float0_exn
  in
  
  let n_cond = Tensor.size conditioning 0 in
  Array.init n_cond (fun i ->
    let w_point = Tensor.slice conditioning ~dim:0 ~start:i ~length:1 in
    compute_weighted_quantile w_point
  ) |> Tensor.of_float1

let compute_sharp_bounds dataset =
  match dataset.common_vars with
  | None -> 0.0  (* No common variables case *)
  | Some w ->
      let ((gamma_y, gamma_d), (nu_y, nu_d)) = 
        decompose_variables dataset.target dataset.features w in
      
      (* First term: gamma terms *)
      let gamma_term = 
        let prod = Tensor.mul gamma_y gamma_d in
        Tensor.mean prod ~dim:[0] |> Tensor.to_float0_exn
      in
      
      (* Second term: conditional quantile functions *)
      let grid_size = 100 in
      let u_grid = Tensor.linspace ~start:0. ~end_:1. grid_size in
      
      let nu_term = Array.init grid_size (fun i ->
        let u = Tensor.get u_grid i |> Tensor.to_float0_exn in
        let f_inv = conditional_quantile_function nu_y w u in
        let g_inv = conditional_quantile_function nu_d w u in
        let prod = Tensor.mul f_inv g_inv in
        Tensor.mean prod ~dim:[0] |> Tensor.to_float0_exn
      ) |> Array.fold_left (+.) 0. |> fun x -> x /. Float.of_int grid_size in
      
      let var_eta = Tensor.var nu_d ~unbiased:true |> Tensor.to_float0_exn in
      (gamma_term +. nu_term) /. var_eta

let compute_eta_d ~dims d x =
  let basis = Array.init (dims.p_o + dims.p_c) (fun i -> 
    if i < dims.p_o + dims.p_c - 1 then
      let v = Tensor.zeros [dims.p_o + dims.p_c] in
      Tensor.fill_float v i 1.0;
      v
    else d
  ) in
  let m_matrix = Tensor.stack (Array.to_list basis) ~dim:0 in
  let t = Tensor.matmul (Tensor.inverse m_matrix) x in
  let t1 = Tensor.select t ~dim:0 ~index:0 in
  let t_rest = Tensor.narrow t ~dim:0 ~start:1 ~length:(dims.p_o + dims.p_c - 1) in
  compute_residuals t_rest t1

let compute_support_function ~dims x y d =
  let eta_d = compute_eta_d ~dims d x in
  let e_eta2 = Tensor.mean (Tensor.mul eta_d eta_d) ~dim:[0] |> Tensor.to_float0_exn in
  
  let compute_quantile_product x y =
    let n = min (Tensor.size x 0) (Tensor.size y 0) in
    let grid = Tensor.linspace ~start:0. ~end_:1. n in
    let x_sorted = Tensor.sort x ~dim:0 ~descending:false |> fst in
    let y_sorted = Tensor.sort y ~dim:0 ~descending:false |> fst in
    Tensor.mean (Tensor.mul x_sorted y_sorted) ~dim:[0] |> Tensor.to_float0_exn
  in
  
  let quantile_prod = compute_quantile_product eta_d y in
  quantile_prod /. e_eta2

let compute_sharp_bounds dataset =
  match dataset.common_vars with
  | None -> 
      compute_support_function 
        ~dims:{p_o=1; p_c=1; q=1} 
        dataset.features dataset.target 
        (Tensor.ones [1])
  | Some _ -> 
      compute_sharp_bounds dataset

let compute_outer_bound dataset g =
  let g_vals = g dataset.features in
  compute_sharp_bounds {dataset with common_vars = Some g_vals}

let compute_influence_functions dataset =
  let eta_d = compute_residuals 
    dataset.features dataset.target in
    
  let e_eta2 = Tensor.mean (Tensor.mul eta_d eta_d) ~dim:[0] 
               |> Tensor.to_float0_exn in

  (* ψ₁: Residual of η_d² regression *)
  let psi1 =
    let eta_squared = Tensor.mul eta_d eta_d in
    Tensor.sub eta_squared (Tensor.full_like eta_squared e_eta2)
    |> Tensor.neg
    |> Tensor.div_scalar e_eta2
  in

  (* ψ₂: Residual from T₁ on T₋₁ regression *)
  let psi2 =
    let t_minus = dataset.features in
    let beta = Tensor.matmul 
      (Tensor.pinverse (Tensor.transpose t_minus ~dim0:0 ~dim1:1)) 
      (Tensor.reshape eta_d ~shape:[-1; 1]) in
    Tensor.matmul t_minus beta
    |> Tensor.squeeze ~dim:[1]
    |> Tensor.neg
    |> Tensor.div_scalar e_eta2
  in

  (* ψ₃: Involves cumulative integrals *)
  let psi3 =
    let n = Tensor.size eta_d 0 in
    let sorted_eta, indices = Tensor.sort eta_d ~dim:0 ~descending:false in
    let grid = Tensor.linspace ~start:0. ~end_:1. n in
    
    Array.init n (fun i ->
      let t = Tensor.get sorted_eta i |> Tensor.to_float0_exn in
      let indicators = Tensor.le eta_d (Tensor.full_like eta_d t) in
      let integral = Tensor.mean (Tensor.mul indicators (Tensor.sub eta_d t)) ~dim:[0] 
                    |> Tensor.to_float0_exn in
      integral
    ) |> Tensor.of_float1
    |> Tensor.neg
  in

  (* ψ₄: Quantile-based terms *)
  let psi4 =
    let n = Tensor.size dataset.target 0 in
    Array.init n (fun i ->
      let y_i = Tensor.get dataset.target i |> Tensor.to_float0_exn in
      let indicators = Tensor.le dataset.target (Tensor.full_like dataset.target y_i) in
      let integral = Tensor.mean (Tensor.mul indicators 
                                 (Tensor.sub dataset.target y_i)) ~dim:[0]
                    |> Tensor.to_float0_exn in
      integral
    ) |> Tensor.of_float1
    |> Tensor.neg
  in

  (psi1, psi2, psi3, psi4)

let compute_asymptotic_variance dataset lambda =
  let (psi1, psi2, psi3, psi4) = compute_influence_functions dataset in
  
  let var tensor = 
    let mean = Tensor.mean tensor ~dim:[0] in
    let centered = Tensor.sub tensor mean in
    Tensor.mean (Tensor.mul centered centered) ~dim:[0]
  in
  
  let v1 = var psi1 |> Tensor.to_float0_exn in
  let v2 = var psi2 |> Tensor.to_float0_exn in
  let v3 = var psi3 |> Tensor.to_float0_exn in
  let v4 = var psi4 |> Tensor.to_float0_exn in
  
  let n = Float.of_int (Tensor.size dataset.target 0) in
  lambda *. (v1 +. v2 +. v3) /. n +. (1. -. lambda) *. v4 /. n

let compute_clt_distribution dataset bound =
  let n = Tensor.size dataset.target 0 in
  let m = Tensor.size dataset.features 0 in
  let lambda = Float.of_int n /. Float.of_int (n + m) in
  
  let var = compute_asymptotic_variance dataset lambda in
  let scale = sqrt (float_of_int (n * m) /. float_of_int (n + m)) in
  
  (bound, scale *. sqrt var)

let confidence_interval ?(alpha=0.05) dataset bound =
  let (mean, std) = compute_clt_distribution dataset bound in
  let z = 1.96 in  (* For 95% confidence *)
  
  let lower = mean -. z *. std in
  let upper = mean +. z *. std in
  (lower, upper)