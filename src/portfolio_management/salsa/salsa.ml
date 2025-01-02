open Torch
open Types
open Matrix_ops
open Stats

let compute_scaling_factors probs sample_size =
  let scaling = Tensor.full [Tensor.shape1_exn probs] (1. /. float_of_int sample_size) in
  Tensor.div scaling (Tensor.sqrt probs)

let rec estimate_leverage_scores_recursive 
    matrix 
    col_idx 
    s1 
    s2 
    prev_scores =
  let m, n = Tensor.shape2_exn matrix in
  
  match col_idx with
  | 1 -> 
      (* Base case: first column *)
      let col = Tensor.select matrix ~dim:1 ~index:0 in
      let norm_sq = Tensor.dot col col in
      Tensor.div (Tensor.mul col col) norm_sq
  | _ ->
      (* Get previous submatrix *)
      let prev_matrix = Tensor.narrow matrix ~dim:1 ~start:0 ~length:(col_idx-1) in
      
      (* Current column *)
      let curr_col = Tensor.select matrix ~dim:1 ~index:(col_idx-1) in
      
      (* Get previous scores or compute them *)
      let prev_levscores = match prev_scores with
        | Some scores -> scores
        | None -> estimate_leverage_scores_recursive prev_matrix (col_idx-1) s1 s2 None
      in
      
      (* Sample rows based on previous scores *)
      let probs = Tensor.div prev_levscores (Tensor.full [m] (float_of_int (col_idx-1))) in
      let sampled_matrix = sample_matrix prev_matrix probs s1 in
      let sampled_col = sample_matrix (Tensor.view curr_col [-1; 1]) probs s1 in
      
      (* Solve least squares on sampled system *)
      let q, r = qr_decomposition sampled_matrix in
      let beta = Tensor.triangular_solve sampled_col r ~upper:true ~transpose:false ~unitriangular:false in
      
      (* Compute residuals *)
      let residuals = compute_residuals prev_matrix beta curr_col in
      
      (* Update leverage scores *)
      let res_norm = Tensor.dot residuals residuals in
      let new_scores = Tensor.div (Tensor.mul residuals residuals) res_norm in
      Tensor.add prev_levscores new_scores

let compute_leverage_scores input_matrix s1 s2 =
  let m, n = Tensor.shape2_exn input_matrix in
  estimate_leverage_scores_recursive input_matrix n s1 s2 None

module OrderSelection = struct
  let rolling_average_selection series max_order window_size =
    let n = Tensor.shape1_exn series in
    let scores = Tensor.zeros [max_order + 1] in
    
    (* Compute BIC scores for different orders *)
    for p = 0 to max_order do
      let window_scores = Tensor.zeros [n - window_size + 1] in
      
      (* Compute BIC for each window *)
      for i = 0 to n - window_size do
        let window = Tensor.narrow series ~dim:0 ~start:i ~length:window_size in
        let ar_params = Lsarma.estimate_ar_params window p in
        let residuals = compute_residuals 
          (Tensor.narrow window ~dim:0 ~start:p ~length:(window_size - p))
          ar_params
          (Tensor.narrow window ~dim:0 ~start:p ~length:(window_size - p)) in
        let bic = Lsarma.compute_bic residuals p in
        Tensor.copy_ 
          (Tensor.narrow window_scores ~dim:0 ~start:i ~length:1)
          (Tensor.full [1] bic)
      done;
      
      (* Store average BIC score for this order *)
      Tensor.copy_
        (Tensor.narrow scores ~dim:0 ~start:p ~length:1)
        (Tensor.mean window_scores |> Tensor.view [-1])
    done;
    
    (* Find optimal order *)
    let _, optimal_order = Tensor.min scores ~dim:0 ~keepdim:false in
    Tensor.item optimal_order |> int_of_float
end

module Lsarma = struct
  let compute_bic residuals num_params =
    let n = float_of_int (Tensor.shape1_exn residuals) in
    let rss = Tensor.sum (Tensor.mul residuals residuals) in
    let sigma2 = Tensor.div rss n in
    let log_likelihood = 
      -0.5 *. n *. (log (2. *. Float.pi) +. 1. +. log (Tensor.item sigma2)) in
    -2. *. log_likelihood +. (float_of_int num_params) *. log n

  let estimate_ar_params time_series ar_order =
    let n = Tensor.shape1_exn time_series in
    let x_matrix = Tensor.zeros [n - ar_order; ar_order] in
    let y_vector = Tensor.narrow time_series ~dim:0 ~start:ar_order ~length:(n - ar_order) in
    
    (* Construct AR matrix *)
    for i = 0 to ar_order - 1 do
      let col = Tensor.narrow time_series ~dim:0 ~start:(ar_order - i - 1) ~length:(n - ar_order) in
      Tensor.copy_ (Tensor.select x_matrix ~dim:1 ~index:i) col
    done;
    
    (* Solve least squares *)
    let q, r = qr_decomposition x_matrix in
    let qty = Tensor.mv (Tensor.transpose q ~dim0:0 ~dim1:1) y_vector in
    Tensor.triangular_solve qty r ~upper:true ~transpose:false ~unitriangular:false

  let estimate_white_noise time_series ar_order =
    let ar_params = estimate_ar_params time_series ar_order in
    let n = Tensor.shape1_exn time_series in
    
    (* Construct prediction matrix *)
    let x_matrix = Tensor.zeros [n - ar_order; ar_order] in
    for i = 0 to ar_order - 1 do
      let col = Tensor.narrow time_series ~dim:0 ~start:(ar_order - i - 1) ~length:(n - ar_order) in
      Tensor.copy_ (Tensor.select x_matrix ~dim:1 ~index:i) col
    done;
    
    (* Compute residuals *)
    let predictions = Tensor.mv x_matrix ar_params in
    let actual = Tensor.narrow time_series ~dim:0 ~start:ar_order ~length:(n - ar_order) in
    let residuals = Tensor.sub actual predictions in
    ar_params, residuals

  let make_stationary time_series max_diff =
    let is_stationary series =
      let acf = compute_acf series 1 in
      Tensor.item (Tensor.get acf [1]) < 0.95 in
      
    let rec transform series diff_count =
      if diff_count >= max_diff then series, diff_count
      else if is_stationary series then series, diff_count
      else
        let n = Tensor.shape1_exn series in
        let diffed = Tensor.sub
          (Tensor.narrow series ~dim:0 ~start:1 ~length:(n-1))
          (Tensor.narrow series ~dim:0 ~start:0 ~length:(n-1)) in
        transform diffed (diff_count + 1) in
    
    transform time_series 0

  let fit_arma time_series p q max_ar_order =
    (* Check and transform for stationarity *)
    let stationary_series, diff_count = make_stationary time_series 2 in
    
    (* Use rolling average for AR order selection if needed *)
    let p = if p = 0 then
      OrderSelection.rolling_average_selection stationary_series max_ar_order 100
    else p in
    
    (* First stage: AR approximation *)
    let ar_params, residuals = estimate_white_noise stationary_series max_ar_order in
    
    (* Construct ARMA data matrix *)
    let n = Tensor.shape1_exn stationary_series in
    let x_matrix = Tensor.zeros [n - max_ar_order - q; p + q] in
    
    (* Fill AR components *)
    for i = 0 to p - 1 do
      let col = Tensor.narrow stationary_series ~dim:0 
        ~start:(max_ar_order + q - i - 1) 
        ~length:(n - max_ar_order - q) in
      Tensor.copy_ (Tensor.select x_matrix ~dim:1 ~index:i) col
    done;
    
    (* Fill MA components *)
    for i = 0 to q - 1 do
      let col = Tensor.narrow residuals ~dim:0 
        ~start:(q - i - 1) 
        ~length:(n - max_ar_order - q) in
      Tensor.copy_ (Tensor.select x_matrix ~dim:1 ~index:(p + i)) col
    done;
    
    (* Response vector *)
    let y = Tensor.narrow stationary_series ~dim:0 
      ~start:(max_ar_order + q) 
      ~length:(n - max_ar_order - q) in
      
    (* Use SALSA to compute leverage scores *)
    let leverage_scores = Salsa.compute_leverage_scores x_matrix (p + q) (p + q / 2) in
    
    (* Sample based on leverage scores and solve *)
    let sample_size = min (5 * (p + q)) (n - max_ar_order - q) in
    let sampled_x = sample_matrix x_matrix leverage_scores sample_size in
    let sampled_y = sample_matrix (Tensor.view y [-1; 1]) leverage_scores sample_size in
    
    (* Solve with regularization *)
    let lambda = 1e-5 *. (Tensor.item (Tensor.mean (Tensor.mul x_matrix x_matrix))) in
    let params = solve_regularized_ls sampled_x sampled_y lambda in
    
    (* Extract coefficients *)
    let ar_coefs = Tensor.narrow params ~dim:0 ~start:0 ~length:p in
    let ma_coefs = Tensor.narrow params ~dim:0 ~start:p ~length:q in
    
    (* Compute final residuals and white noise variance *)
    let final_residuals = compute_residuals x_matrix params y in
    let white_noise_var = Tensor.item (Tensor.mean (Tensor.mul final_residuals final_residuals)) in
    
    (* Compute standard errors *)
    let xtx = Tensor.mm (Tensor.transpose x_matrix ~dim0:0 ~dim1:1) x_matrix in
    let xtx_inv = Tensor.inverse xtx in
    let mse = Tensor.item (Tensor.mean (Tensor.mul final_residuals final_residuals)) in
    let std_errors = Tensor.sqrt (Tensor.mul_scalar (Tensor.diag xtx_inv) mse) in
    
    (* Compute BIC *)
    let bic = compute_bic final_residuals (p + q) in
    
    {ar_coefs; ma_coefs; white_noise_var; bic_score = bic; std_errors; residuals = final_residuals}

  let compute_diagnostics params time_series =
    let n = float_of_int (Tensor.shape1_exn time_series) in
    let k = float_of_int (Tensor.shape1_exn params.ar_coefs + Tensor.shape1_exn params.ma_coefs) in
    
    let log_likelihood = 
      -0.5 *. n *. (log (2. *. Float.pi) +. 1. +. log params.white_noise_var) in
      
    let aic = -2. *. log_likelihood +. 2. *. k in
    let residual_acf = compute_acf params.residuals 20 in
    let ljung_box_stat = ljung_box_test params.residuals 20 in
    let durbin_watson = durbin_watson params.residuals in
    
    {
      aic;
      bic = params.bic_score;
      log_likelihood;
      durbin_watson;
      ljung_box_stat;
      residual_acf;
    }

  let parallel_select_order time_series max_p max_q max_ar_order =
    let num_models = (max_p + 1) * (max_q + 1) - 1 in
    let models = Array.make num_models (0, 0) in
    let idx = ref 0 in
    for p = 0 to max_p do
      for q = 0 to max_q do
        if p + q > 0 then begin
          models.(!idx) <- (p, q);
          incr idx
        end
      done
    done;
    
    let num_domains = min (Domain.recommended_domain_count ()) num_