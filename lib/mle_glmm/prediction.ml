open Types
open Torch
open MatrixOps

(* BLUP (Best Linear Unbiased Prediction) *)
let predict_random_effects model state =
  let d = Covariance.compute_covariance model.spec 
            (SpatialUtils.distance_matrix model.z) in
  
  let r = Glmm.compute_r model state.w (Tensor.zeros [2; 1]) in
  let r_inv = safe_inverse r in
  
  (* v_βω,00 : D Z' R⁻¹(y_tilde - Xβ) *)
  let term1 = Tensor.mm d (Tensor.transpose model.z ~dim0:0 ~dim1:1) in
  let diff = Tensor.sub state.y_tilde 
              (Tensor.mm model.x state.beta) in
  let term2 = Tensor.mm r_inv diff in
  Tensor.mm term1 term2

(* Compute prediction variance (V_βω,0) *)
let prediction_variance model gamma =
  let (_, w) = Glmm.compute_working_response model gamma in
  
  let d = Covariance.compute_covariance model.spec 
            (SpatialUtils.distance_matrix model.z) in
  let r = Glmm.compute_r model w (Tensor.zeros [2; 1]) in
  let r_inv = safe_inverse r in
  
  (* V_βω,0 = D - D Z' R⁻¹ Z D *)
  let term1 = d in
  let term2 = Tensor.mm d (Tensor.transpose model.z ~dim0:0 ~dim1:1) in
  let term3 = Tensor.mm r_inv model.z in
  let term4 = Tensor.mm term2 term3 in
  Tensor.sub term1 term4

let predict_response model state new_x new_z =
  (* Compute fixed effects contribution *)
  let fixed = Tensor.mm new_x state.beta in
  
  (* Compute random effects contribution *)
  let random = Tensor.mm new_z state.gamma in
  
  (* Combine fixed and random effects *)
  let eta = Tensor.add fixed random in
  
  (* Apply inverse link function based on distribution *)
  match model.spec.distribution with
  | Binomial _ -> Tensor.sigmoid eta
  | Poisson -> Tensor.exp eta
  | Normal _ -> eta

let prediction_intervals model state new_x new_z alpha =
  let pred = predict_response model state new_x new_z in
  
  let pred_var = prediction_variance model state.gamma in
  
  let z = Scalar.float (Stats.Normal.ppf ((1.0 +. alpha) /. 2.0)) in
  let margin = Tensor.sqrt pred_var |>
               Tensor.mul_scalar z in
  
  let total_var = match model.spec.distribution with
    | Binomial _ ->
        let p = pred in
        Tensor.mul p (Tensor.sub (Tensor.ones_like p) p)
    | Poisson ->
        pred
    | Normal {variance} ->
        Tensor.ones_like pred |>
        Tensor.mul_scalar (Scalar.float variance) in
  
  let total_margin = Tensor.sqrt (Tensor.add (Tensor.pow margin 
                                              (Tensor.scalar_float 2.0))
                                            total_var) in
  
  let lower = Tensor.sub pred total_margin in
  let upper = Tensor.add pred total_margin in
  (lower, upper)

(* K-fold cross-validation *)
let cross_validate model k =
  let n = model.spec.n_obs in
  let fold_size = n / k in
  
  let errors = Array.make k 0.0 in
  
  for i = 0 to k - 1 do
    let start_idx = i * fold_size in
    let end_idx = min (n - 1) ((i + 1) * fold_size - 1) in
    
    let train_indices = Tensor.cat [
      Tensor.arange ~start:0 ~end_:start_idx;
      Tensor.arange ~start:(end_idx + 1) ~end_:n
    ] ~dim:0 in
    
    let val_indices = Tensor.arange ~start:start_idx ~end_:(end_idx + 1) in
    
    let train_x = Tensor.index_select model.x ~dim:0 ~index:train_indices in
    let train_z = Tensor.index_select model.z ~dim:0 ~index:train_indices in
    let train_y = Tensor.index_select model.y ~dim:0 ~index:train_indices in
    
    let val_x = Tensor.index_select model.x ~dim:0 ~index:val_indices in
    let val_z = Tensor.index_select model.z ~dim:0 ~index:val_indices in
    let val_y = Tensor.index_select model.y ~dim:0 ~index:val_indices in
    
    let train_model = Glmm.create model.spec train_x train_z train_y in
    let state = Algorithm1.run train_model in
    
    let pred_y = predict_response train_model state val_x val_z in
    
    let error = match model.spec.distribution with
      | Binomial _ ->
          let bce = Tensor.binary_cross_entropy pred_y val_y in
          Tensor.scalar_to_float bce |> Scalar.to_float
      | Poisson ->
          let mse = Tensor.mse_loss pred_y val_y in
          Tensor.scalar_to_float mse |> Scalar.to_float
      | Normal _ ->
          let mse = Tensor.mse_loss pred_y val_y in
          Tensor.scalar_to_float mse |> Scalar.to_float in
    
    errors.(i) <- error
  done;
  
  let mean_error = Array.fold_left (+.) 0.0 errors /. float k in
  let sq_diff = Array.map (fun x -> (x -. mean_error) ** 2.0) errors in
  let std_error = sqrt (Array.fold_left (+.) 0.0 sq_diff /. float (k - 1)) in
  
  (mean_error, std_error)

let predict_conditional model state new_x new_z new_y =
  let pred = predict_response model state new_x new_z in
  
  let eta_new = Tensor.add (Tensor.mm new_x state.beta)
                          (Tensor.mm new_z state.gamma) in
  
  let (y_tilde, w) = match model.spec.distribution with
    | Binomial {trials} ->
        let p = Tensor.sigmoid eta_new in
        let w = Tensor.mul trials (Tensor.mul p (Tensor.sub (Tensor.ones_like p) p)) in
        let y_t = Tensor.add eta_new 
                   (Tensor.div (Tensor.sub new_y (Tensor.mul trials p)) w) in
        (y_t, w)
    | Poisson ->
        let mu = Tensor.exp eta_new in
        let w = mu in
        let y_t = Tensor.add eta_new 
                   (Tensor.div (Tensor.sub new_y mu) w) in
        (y_t, w)
    | Normal {variance} ->
        let w = Tensor.ones_like eta_new |> 
                Tensor.mul_scalar (Scalar.float (1.0 /. variance)) in
        (new_y, w) in
  
  let r = Glmm.compute_r model w (Tensor.zeros [2; 1]) in
  let r_inv = safe_inverse r in
  
  let diff = Tensor.sub y_tilde pred in
  let update = Tensor.mm (Tensor.mm (Tensor.transpose new_z ~dim0:0 ~dim1:1)
                                   r_inv)
                        diff in
  
  Tensor.add pred update