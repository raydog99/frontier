open Torch
open Utils

let conv_transition alpha q =
  let scale = Tensor.scalar_tensor (1. /. (2. *. alpha)) in
  let identity = Tensor.eye (Tensor.size q).[0] in
  Tensor.(q + scale * identity)
  
let conv_measurement beta r = 
  let scale = Tensor.scalar_tensor (1. /. (2. *. beta)) in
  let identity = Tensor.eye (Tensor.size r).[0] in
  Tensor.(r + scale * identity)

(* Linear Kalman Filter *)
module LinearKalman = struct
  type kalman_state = {
    mean: Tensor.t;
    covar: Tensor.t;
  }

  let predict_step alpha state transition_mat process_noise =
    let q_conv = conv_transition alpha process_noise in
    let mean = Tensor.(mm transition_mat state.mean) in
    let covar = Tensor.(
      mm (mm transition_mat state.covar) (transpose transition_mat ~dim0:0 ~dim1:1) 
      + q_conv
    ) |> ensure_pos_def in
    { mean; covar }

  let update_step beta state measurement_mat measurement noise =
    let r_conv = conv_measurement beta noise in
    let innovation = Tensor.(
      measurement - mm measurement_mat state.mean
    ) in
    let s = Tensor.(
      mm (mm measurement_mat state.covar) (transpose measurement_mat ~dim0:0 ~dim1:1) 
      + r_conv
    ) |> ensure_pos_def in
    let k = Tensor.(
      mm (mm state.covar (transpose measurement_mat ~dim0:0 ~dim1:1)) (inverse s)
    ) in
    let mean = Tensor.(
      state.mean + mm k innovation
    ) in
    let covar = Tensor.(
      state.covar - mm (mm k measurement_mat) state.covar
    ) |> ensure_pos_def in
    { mean; covar }
end

(* Extended Kalman Filter *)
module ExtendedKalman = struct
  type ekf_state = {
    mean: Tensor.t;
    covar: Tensor.t;
  }

  let linearize_transition transition_fn state params epsilon =
    let n = (Tensor.size state.mean).[0] in
    let f_x = transition_fn state.mean in
    let jacobian = Tensor.empty [n; n] in
    
    for i = 0 to n - 1 do
      let perturbed = Tensor.copy state.mean in
      Tensor.set perturbed [i] Tensor.(get state.mean [i] + epsilon);
      let f_x_perturbed = transition_fn perturbed in
      let diff = Tensor.((f_x_perturbed - f_x) / epsilon) in
      for j = 0 to n - 1 do
        Tensor.set jacobian [i; j] (Tensor.get diff [j])
      done
    done;
    jacobian, f_x

  let predict_step alpha state transition_fn process_noise params epsilon =
    let jacobian, predicted_mean = linearize_transition transition_fn state params epsilon in
    let q_conv = conv_transition alpha process_noise in
    let covar = Tensor.(
      mm (mm jacobian state.covar) (transpose jacobian ~dim0:0 ~dim1:1) 
      + q_conv
    ) |> ensure_pos_def in
    { mean = predicted_mean; covar }

  let update_step beta state measurement_fn measurement noise params epsilon =
    let h_jacobian, predicted_measurement = 
      linearize_transition measurement_fn state params epsilon in
    let r_conv = conv_measurement beta noise in
    let innovation = Tensor.(measurement - predicted_measurement) in
    let s = Tensor.(
      mm (mm h_jacobian state.covar) (transpose h_jacobian ~dim0:0 ~dim1:1) 
      + r_conv
    ) |> ensure_pos_def in
    let k = Tensor.(
      mm (mm state.covar (transpose h_jacobian ~dim0:0 ~dim1:1)) (inverse s)
    ) in
    let mean = Tensor.(state.mean + mm k innovation) in
    let covar = Tensor.(
      state.covar - mm (mm k h_jacobian) state.covar
    ) |> ensure_pos_def in
    { mean; covar }
end

(* Unscented Kalman Filter *)
module UnscentedKalman = struct
  type ukf_state = {
    mean: Tensor.t;
    covar: Tensor.t;
  }

  let generate_sigma_points state lambda =
    let n = (Tensor.size state.mean).[0] in
    let scaled_covar = Tensor.(scalar_tensor (float_of_int n +. lambda) * state.covar) in
    let l = Tensor.cholesky scaled_covar in
    
    let sigma_points = Tensor.empty [2 * n + 1; n] in
    Tensor.copy_ (Tensor.select sigma_points ~dim:0 ~index:0) state.mean;
    
    for i = 0 to n - 1 do
      let li = Tensor.select l ~dim:1 ~index:i in
      Tensor.copy_ 
        (Tensor.select sigma_points ~dim:0 ~index:(i + 1)) 
        Tensor.(state.mean + li);
      Tensor.copy_ 
        (Tensor.select sigma_points ~dim:0 ~index:(n + i + 1)) 
        Tensor.(state.mean - li)
    done;
    sigma_points

  let compute_weights n lambda =
    let w0_m = lambda /. (float_of_int n +. lambda) in
    let w0_c = w0_m +. (1. -. lambda *. lambda /. (float_of_int n +. lambda)) in
    let wi = 1. /. (2. *. (float_of_int n +. lambda)) in
    w0_m, w0_c, wi

  let predict_step alpha state transition_fn process_noise params lambda =
    let n = (Tensor.size state.mean).[0] in
    let sigma_points = generate_sigma_points state lambda in
    let w0_m, w0_c, wi = compute_weights n lambda in
    
    (* Transform sigma points *)
    let transformed_sigmas = Tensor.empty (Tensor.size sigma_points) in
    for i = 0 to 2 * n do
      let sigma = Tensor.select sigma_points ~dim:0 ~index:i in
      let transformed = transition_fn sigma params in
      Tensor.copy_ (Tensor.select transformed_sigmas ~dim:0 ~index:i) transformed
    done;
    
    (* Compute predicted mean *)
    let mean = Tensor.(
      scalar_tensor w0_m * (select transformed_sigmas ~dim:0 ~index:0) +
      scalar_tensor wi * (sum transformed_sigmas ~dim:0)
    ) in
    
    (* Compute predicted covariance *)
    let q_conv = conv_transition alpha process_noise in
    let covar = ref q_conv in
    
    for i = 0 to 2 * n do
      let weight = if i = 0 then w0_c else wi in
      let diff = Tensor.(select transformed_sigmas ~dim:0 ~index:i - mean) in
      covar := Tensor.(
        !covar + 
        scalar_tensor weight * 
        mm (unsqueeze diff ~dim:1) (unsqueeze diff ~dim:0)
      )
    done;
    
    { mean; covar = ensure_pos_def !covar }

  let update_step beta state measurement_fn measurement noise params lambda =
    let n = (Tensor.size state.mean).[0] in
    let m = (Tensor.size measurement).[0] in
    let sigma_points = generate_sigma_points state lambda in
    let w0_m, w0_c, wi = compute_weights n lambda in
    
    (* Transform sigma points through measurement function *)
    let transformed_sigmas = Tensor.empty [2 * n + 1; m] in
    for i = 0 to 2 * n do
      let sigma = Tensor.select sigma_points ~dim:0 ~index:i in
      let transformed = measurement_fn sigma params in
      Tensor.copy_ (Tensor.select transformed_sigmas ~dim:0 ~index:i) transformed
    done;
    
    (* Predicted measurement *)
    let pred_measurement = Tensor.(
      scalar_tensor w0_m * (select transformed_sigmas ~dim:0 ~index:0) +
      scalar_tensor wi * (sum transformed_sigmas ~dim:0)
    ) in
    
    (* Innovation covariance *)
    let r_conv = conv_measurement beta noise in
    let s = ref r_conv in
    let p_xy = ref (Tensor.zeros [n; m]) in
    
    for i = 0 to 2 * n do
      let weight = if i = 0 then w0_c else wi in
      let x_diff = Tensor.(select sigma_points ~dim:0 ~index:i - state.mean) in
      let y_diff = Tensor.(
        select transformed_sigmas ~dim:0 ~index:i - pred_measurement
      ) in
      
      s := Tensor.(
        !s + 
        scalar_tensor weight * 
        mm (unsqueeze y_diff ~dim:1) (unsqueeze y_diff ~dim:0)
      );
      
      p_xy := Tensor.(
        !p_xy + 
        scalar_tensor weight * 
        mm (unsqueeze x_diff ~dim:1) (unsqueeze y_diff ~dim:0)
      )
    done;
    
    let s = ensure_pos_def !s in
    let k = Tensor.(mm !p_xy (inverse s)) in
    
    let innovation = Tensor.(measurement - pred_measurement) in
    let mean = Tensor.(state.mean + mm k innovation) in
    let covar = Tensor.(
      state.covar - mm (mm k s) (transpose k ~dim0:0 ~dim1:1)
    ) |> ensure_pos_def in
    
    { mean; covar }

  let create_estimator state =
    (module struct
      type state = ukf_state
      let estimate_state s = s.mean
      let estimate_uncertainty s = Some s.covar
    end : Types.StateEstimator with type state = ukf_state)
end