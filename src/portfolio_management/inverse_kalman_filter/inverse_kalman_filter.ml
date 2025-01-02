open Torch

module Types = struct
  type state = {
    mean: Tensor.t;
    covariance: Tensor.t;
  }

  type observation = {
    value: Tensor.t;
    variance: Tensor.t;
  }

  type dlm_params = {
    ft: Tensor.t;
    gt: Tensor.t;
    wt: Tensor.t;
    vt: Tensor.t;
    bt: Tensor.t;
    qt: Tensor.t;
    kt: Tensor.t;
  }

  type model_params = {
    gamma: Tensor.t;
    eta: Tensor.t;
    sigma2_0: float;
  }

  type training_data = {
    inputs: Tensor.t;
    outputs: Tensor.t;
    num_states: int;
    num_obs: int;
  }

  type evaluation_metrics = {
    nrmse: float;
    credible_interval_length: float;
    coverage_proportion: float;
  }
end

module KalmanFilter = struct
  open Types

  let compute_filter_matrices ft gt wt vt state_dim obs_dim =
    let rec loop t acc =
      if t = 0 then acc
      else
        let prev = List.hd acc in
        let bt = Tensor.(add (mm (mm gt prev.bt) (transpose gt ~dim0:0 ~dim1:1)) wt) in
        let qt = Tensor.(add (mm (mm ft bt) (transpose ft ~dim0:0 ~dim1:1)) vt) in
        let kt = Tensor.(mm (mm bt (transpose ft ~dim0:0 ~dim1:1)) (inverse qt)) in
        let params = {ft; gt; wt; vt; bt; qt; kt} in
        loop (t-1) (params :: acc)
    in
    let init_params = {
      ft = Tensor.eye state_dim;
      gt = Tensor.eye state_dim;
      wt = Tensor.eye state_dim;
      vt = Tensor.eye obs_dim;
      bt = Tensor.eye state_dim;
      qt = Tensor.eye obs_dim;
      kt = Tensor.eye state_dim;
    } in
    loop state_dim [init_params]
end

module InverseKalmanFilter = struct
  open Types

  let compute_lt_transpose_u params_list u =
    let n = List.length params_list in
    let x_tilde = Tensor.zeros [n] in
    
    let params_n = List.nth params_list (n-1) in
    let params_n1 = List.nth params_list (n-2) in
    
    let x_tilde_n = Tensor.(sqrt params_n.qt * (get u [n-1])) in
    let g_n1 = Tensor.(mm params_n.ft params_n.gt * (get u [n-1])) in
    let x_tilde_n1 = Tensor.(
      sqrt params_n1.qt * 
      (add 
        (mm params_n.ft params_n.gt params_n1.kt * (get u [n-1])) 
        (get u [n-2]))
    ) in
    
    let rec compute_elements t g_t x_tilde_acc =
      if t < 0 then x_tilde_acc
      else
        let params_t = List.nth params_list t in
        let params_next = List.nth params_list (t+1) in
        
        let l_next_t = Tensor.(mm params_next.ft params_next.gt params_t.kt) in
        let l_tilde_next_t = Tensor.(mm g_t params_next.gt params_t.kt) in
        
        let g_t_new = Tensor.(
          add
            (mm g_t params_next.gt)
            (mm params_next.ft params_next.gt * (get u [t+1]))
        ) in
        
        let x_tilde_t = Tensor.(
          sqrt params_t.qt * 
          (add
            (add l_tilde_next_t (mul l_next_t (get u [t+1])))
            (get u [t]))
        ) in
        
        compute_elements (t-1) g_t_new (x_tilde_t :: x_tilde_acc)
    in
    
    compute_elements (n-3) g_n1 [x_tilde_n1; x_tilde_n]

  let compute_l_x_tilde params_list x_tilde =
    let n = List.length params_list in
    let params_1 = List.hd params_list in
    
    let x1 = Tensor.(
      add
        (mm params_1.ft params_1.bt)
        (sqrt params_1.qt * (get x_tilde [0]))
    ) in
    let m1 = Tensor.(
      add params_1.bt
        (mm params_1.kt (sub x1 (mm params_1.ft params_1.bt)))
    ) in
    
    let rec compute_elements t m_prev x_acc =
      if t >= n then x_acc
      else
        let params_t = List.nth params_list t in
        
        let bt = Tensor.(mm params_t.gt m_prev) in
        let xt = Tensor.(
          add
            (mm params_t.ft bt)
            (sqrt params_t.qt * (get x_tilde [t]))
        ) in
        let mt = Tensor.(
          add bt
            (mm params_t.kt (sub xt (mm params_t.ft bt)))
        ) in
        
        compute_elements (t+1) mt (xt :: x_acc)
    in
    
    compute_elements 1 m1 [x1]

  let compute_sigma_u params_list u =
    let x_tilde = compute_lt_transpose_u params_list u in
    compute_l_x_tilde params_list x_tilde

  let compute_sigma_u_robust params_list u eps =
    let make_robust params =
      let qt_robust = Tensor.(add params.qt (eye_like params.qt * eps)) in
      {params with qt = qt_robust}
    in
    let robust_params = List.map make_robust params_list in
    let x = compute_sigma_u robust_params u in
    Tensor.(sub x (mul u eps))

  let compute_lt_inv_x_tilde params_list x_tilde =
    let n = List.length params_list in
    
    let params_n = List.nth params_list (n-1) in
    let u_n = Tensor.(div (get x_tilde [n-1]) (sqrt params_n.qt)) in
    
    let params_n1 = List.nth params_list (n-2) in
    let l_n_n1 = Tensor.(mm params_n.ft params_n.gt params_n1.kt) in
    let u_n1 = Tensor.(
      div (get x_tilde [n-2]) (sqrt params_n1.qt) - 
      mul l_n_n1 u_n
    ) in
    
    let rec compute_entries t u_acc =
      if t < 0 then u_acc
      else
        let params_t = List.nth params_list t in
        let params_next = List.nth params_list (t+1) in
        
        let l_next_t = Tensor.(mm params_next.ft params_next.gt params_t.kt) in
        let l_tilde_next_t = Tensor.(mm (get x_tilde [t+1]) params_next.gt params_t.kt) in
        
        let u_t = Tensor.(
          div (get x_tilde [t]) (sqrt params_t.qt) -
          sub l_tilde_next_t (mul l_next_t (List.hd u_acc))
        ) in
        
        compute_entries (t-1) (u_t :: u_acc)
    in
    
    compute_entries (n-3) [u_n1; u_n]
end

module ConjugateGradient = struct
  let solve matrix_vector_prod b max_iter tol =
    let x = Tensor.zeros_like b in
    let r = Tensor.copy b in
    let p = Tensor.copy b in
    
    let rec iterate k x r p =
      if k >= max_iter then x
      else
        let ap = matrix_vector_prod p in
        let alpha = Tensor.(
          div
            (dot r r)
            (dot p ap)
        ) in
        let x_next = Tensor.(add x (mul alpha p)) in
        let r_next = Tensor.(sub r (mul alpha ap)) in
        
        if Tensor.((dot r_next r_next) < tol) then x_next
        else
          let beta = Tensor.(
            div
              (dot r_next r_next)
              (dot r r)
          ) in
          let p_next = Tensor.(add r_next (mul beta p)) in
          iterate (k+1) x_next r_next p_next
    in
    
    iterate 0 x r p
end

module IkfCg = struct
  open Types

  let matvec params_list x =
    InverseKalmanFilter.compute_sigma_u params_list x

  let predict_mean params_list observation =
    let matrix_vector_prod = matvec params_list in
    let max_iter = 100 in
    let tol = 1e-6 in
    
    let u = ConjugateGradient.solve matrix_vector_prod observation.value max_iter tol in
    let sigma_u = matrix_vector_prod u in
    sigma_u

  let predict_variance params_list d_star =
    let n = List.length params_list in
    let params_1 = List.hd params_list in
    
    let sigma_d_star = InverseKalmanFilter.compute_sigma_u params_list d_star in
    let c_star = Tensor.(mm params_1.ft params_1.bt) in
    
    Tensor.(sub c_star (dot sigma_d_star sigma_d_star))
end

module ParameterEstimation = struct
  open Types

  let approximate_log_det sigma_0 n0 m0 =
    let omega = Tensor.randn [n0] in
    
    let rec power_iteration sigma_0 omega m =
      if m = 0 then omega
      else
        let next = Tensor.mm sigma_0 omega in
        power_iteration sigma_0 next (m-1)
    in
    
    let sigma_m0_omega = power_iteration sigma_0 omega m0 in
    let q, r = Tensor.qr sigma_m0_omega in
    let sigma_a = Tensor.mm (Tensor.mm (Tensor.transpose q ~dim0:0 ~dim1:1) sigma_0) q in
    Tensor.logdet (Tensor.add sigma_a (Tensor.eye n0))

  let cross_validate model_params train_data val_data =
    let {gamma; eta; sigma2_0} = model_params in
    let {inputs=train_x; outputs=train_y} = train_data in
    let {inputs=val_x; outputs=val_y} = val_data in
    
    let params_list = KalmanFilter.compute_filter_matrices 
      train_data.num_states train_data.num_obs gamma eta sigma2_0 in
    
    let pred_y = IkfCg.predict_mean params_list 
      {value=val_y; variance=Tensor.full_like val_y sigma2_0} in
    
    Tensor.(mean (pow (sub pred_y val_y) (Scalar 2.)))

  let maximum_likelihood model_params data =
    let {gamma; eta; sigma2_0} = model_params in
    let {inputs; outputs; num_states;

module MaternKernel = struct
  let matern_to_dlm nu sigma2 gamma =
    match nu with
    | 0.5 -> 
      (* Exponential kernel - first order DLM *)
      let ft = Tensor.ones [1; 1] in
      let gt = Tensor.full [1; 1] (exp (-1. /. gamma)) in
      let wt = Tensor.full [1; 1] (sigma2 *. (1. -. exp (-2. /. gamma))) in
      {Types.ft; gt; wt; vt = Tensor.zeros [1; 1]}
      
    | 2.5 ->
      (* Third order DLM *)
      let lambda = sqrt(5.) /. gamma in
      let ft = Tensor.cat [Tensor.ones [1; 1]; Tensor.zeros [1; 2]] ~dim:1 in
      let gt = Tensor.cat [
        Tensor.cat [Tensor.ones [1; 1]; Tensor.ones [1; 1]; Tensor.zeros [1; 1]] ~dim:1;
        Tensor.cat [Tensor.zeros [1; 1]; Tensor.ones [1; 1]; Tensor.ones [1; 1]] ~dim:1;
        Tensor.cat [Tensor.zeros [1; 1]; Tensor.zeros [1; 1]; Tensor.full [1; 1] (-lambda)] ~dim:1
      ] ~dim:0 in
      let wt = sigma2 *. (1. -. exp (-2. *. lambda)) *. Tensor.eye 3 in
      {Types.ft; gt; wt; vt = Tensor.zeros [1; 1]}
      
    | _ -> failwith "Only nu = 0.5 and 2.5 are supported"
end

module Evaluation = struct
  let compute_nrmse pred_values true_values =
    let diff = Tensor.(pred_values - true_values) in
    let mse = Tensor.(mean (pow diff (Scalar 2.))) in
    
    let mean_true = Tensor.mean true_values in
    let var_true = Tensor.(mean (pow (true_values - mean_true) (Scalar 2.))) in
    
    Tensor.(sqrt (div mse var_true))
    |> Tensor.float_value

  let compute_interval_length pred_variance =
    let z_score = 1.96 in  (* 95% CI for normal distribution *)
    let interval_lengths = Tensor.(mul (sqrt pred_variance) (Scalar (2. *. z_score))) in
    Tensor.mean interval_lengths
    |> Tensor.float_value

  let compute_coverage pred_values pred_variance true_values =
    let z_score = 1.96 in
    let std_dev = Tensor.sqrt pred_variance in
    let lower = Tensor.(sub pred_values (mul (Scalar z_score) std_dev)) in
    let upper = Tensor.(add pred_values (mul (Scalar z_score) std_dev)) in
    
    let covered = Tensor.(
      logical_and 
        (ge true_values lower)
        (le true_values upper)
    ) in
    
    Tensor.mean covered
    |> Tensor.float_value

  let evaluate pred_values pred_variance true_values =
    {
      Types.nrmse = compute_nrmse pred_values true_values;
      credible_interval_length = compute_interval_length pred_variance;
      coverage_proportion = compute_coverage pred_values pred_variance true_values;
    }
end