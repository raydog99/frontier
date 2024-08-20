open Torch
open Owl
open Owl_plplot
open Parany

type frequency = Daily | Weekly | Monthly | Quarterly

type observation = {
  value : float;
  frequency : frequency;
  date : float; 
}

type model_type =
  | Static
  | Dynamic of int
  | FAVAR of int (* Factor-Augmented VAR *)

type t = {
  n : int;
  r : int;
  t : int;
  model_type : model_type;
  b : Tensor.t;
  f : Tensor.t;
  epsilon : Tensor.t;
  var_coef : Tensor.t option; (* For FAVAR models *)
}

let create n r t model_type =
  let b = match model_type with
    | Static -> Tensor.randn [n; r]
    | Dynamic lags -> Tensor.randn [n; r * (lags + 1)]
    | FAVAR _ -> Tensor.randn [n; r]
  in
  let f = Tensor.randn [r; t] in
  let epsilon = Tensor.randn [n; t] in
  { n; r; t; model_type; b; f; epsilon; var_coef = None }

let create_favar n r t p =
  let b = Tensor.randn [n; r] in
  let f = Tensor.randn [r; t] in
  let epsilon = Tensor.randn [n; t] in
  let var_coef = Tensor.randn [r; r * p] in
  { n; r; t; model_type = FAVAR p; b; f; epsilon; var_coef = Some var_coef }

let forward model x =
  match model.model_type with
  | Static ->
      let common = Tensor.mm model.b model.f in
      Tensor.add common model.epsilon
  | Dynamic lags ->
      let padded_f = Tensor.constant_pad_nd model.f ~pad:[lags; 0; 0; 0] in
      let common = Tensor.fold_left (fun acc i ->
        let fi = Tensor.narrow padded_f ~dim:1 ~start:i ~length:model.t in
        let bi = Tensor.narrow model.b ~dim:1 ~start:(i * model.r) ~length:model.r in
        Tensor.add acc (Tensor.mm bi fi)
      ) (Tensor.zeros [model.n; model.t]) (List.init (lags + 1) (fun x -> x))
      in
      Tensor.add common model.epsilon
  | FAVAR p ->
      let common = Tensor.mm model.b model.f in
      let var_part = match model.var_coef with
        | Some coef ->
            let padded_f = Tensor.constant_pad_nd model.f ~pad:[p; 0; 0; 0] in
            Tensor.fold_left (fun acc i ->
              let fi = Tensor.narrow padded_f ~dim:1 ~start:i ~length:model.t in
              let coef_i = Tensor.narrow coef ~dim:1 ~start:(i * model.r) ~length:model.r in
              Tensor.add acc (Tensor.mm coef_i fi)
            ) (Tensor.zeros [model.r; model.t]) (List.init p (fun x -> x))
        | None -> Tensor.zeros [model.r; model.t]
      in
      Tensor.add (Tensor.add common var_part) model.epsilon

let estimate_factors_pca model x =
  let x_centered = Tensor.sub x (Tensor.mean x ~dim:[1] ~keepdim:true) in
  let cov = Tensor.mm (Tensor.transpose x_centered ~dim0:0 ~dim1:1) x_centered in
  let cov = Tensor.div cov (Float.of_int model.t |> Tensor.of_float) in
  let u, _, _ = Tensor.svd cov ~some:false in
  Tensor.narrow u ~dim:1 ~start:0 ~length:model.r

let estimate_loadings_ols model x factors =
  let x_centered = Tensor.sub x (Tensor.mean x ~dim:[1] ~keepdim:true) in
  let factors_t = Tensor.transpose factors ~dim0:0 ~dim1:1 in
  let inv_factor_cov = Tensor.pinverse (Tensor.mm factors factors_t) in
  Tensor.mm (Tensor.mm x_centered factors_t) inv_factor_cov

let qmle_objective model x =
  let residuals = Tensor.sub x (forward model x) in
  let sigma = Tensor.mm (Tensor.transpose residuals ~dim0:0 ~dim1:1) residuals in
  let sigma = Tensor.div sigma (Float.of_int model.t |> Tensor.of_float) in
  let log_det = Tensor.logdet sigma in
  let trace = Tensor.trace (Tensor.mm (Tensor.inverse sigma) (Tensor.mm (Tensor.transpose x ~dim0:0 ~dim1:1) x)) in
  Tensor.add log_det trace

let estimate_qmle model x max_iter learning_rate =
  let rec optimize iter model =
    if iter >= max_iter then model
    else
      let loss = qmle_objective model x in
      let grad_b = Tensor.grad loss ~inputs:[model.b] |> List.hd in
      let grad_f = Tensor.grad loss ~inputs:[model.f] |> List.hd in
      let grad_var_coef = match model.var_coef with
        | Some coef -> Tensor.grad loss ~inputs:[coef] |> List.hd
        | None -> Tensor.zeros_like model.b
      in
      let new_b = Tensor.sub model.b (Tensor.mul grad_b learning_rate) in
      let new_f = Tensor.sub model.f (Tensor.mul grad_f learning_rate) in
      let new_var_coef = match model.var_coef with
        | Some coef -> Some (Tensor.sub coef (Tensor.mul grad_var_coef learning_rate))
        | None -> None
      in
      optimize (iter + 1) { model with b = new_b; f = new_f; var_coef = new_var_coef }
  in
  optimize 0 model

let fit model x max_iter method_ =
  match method_ with
  | `PCA ->
      let rec fit_iter model iter =
        if iter >= max_iter then model
        else
          let factors = estimate_factors_pca model x in
          let loadings = estimate_loadings_ols model x factors in
          let new_model = { model with b = loadings; f = factors } in
          fit_iter new_model (iter + 1)
      in
      fit_iter model 0
  | `QMLE learning_rate ->
      estimate_qmle model x max_iter learning_rate
  | `FAVAR learning_rate ->
      estimate_qmle model x max_iter learning_rate

let estimate_var_coefficients factors p =
  let t = Tensor.size factors 1 in
  let y = Tensor.narrow factors ~dim:1 ~start:p ~length:(t - p) in
  let x = Tensor.cat (List.init p (fun i ->
    Tensor.narrow factors ~dim:1 ~start:(p - i - 1) ~length:(t - p)
  )) ~dim:0 in
  let x_t = Tensor.transpose x ~dim0:0 ~dim1:1 in
  let inv_xx = Tensor.pinverse (Tensor.mm x_t x) in
  Tensor.mm (Tensor.mm inv_xx x_t) y

let forecast_var model x h =
  let factors = estimate_factors_pca model x in
  let p = 1 in
  let coeffs = estimate_var_coefficients factors p in
  let last_factors = Tensor.narrow factors ~dim:1 ~start:(Tensor.size factors 1 - p) ~length:p in
  let rec forecast_iter current_factors h_left acc =
    if h_left = 0 then List.rev acc
    else
      let next_factor = Tensor.mm coeffs current_factors in
      let new_factors = Tensor.cat [Tensor.narrow current_factors ~dim:0 ~start:1 ~length:(p-1); next_factor] ~dim:0 in
      forecast_iter new_factors (h_left - 1) (next_factor :: acc)
  in
  let future_factors = forecast_iter last_factors h [] |> Tensor.cat ~dim:1 in
  Tensor.mm model.b future_factors

let handle_missing_data x =
  let mask = Tensor.isnan x |> Tensor.logical_not in
  let x_filled = Tensor.where mask x (Tensor.zeros_like x) in
  let row_means = Tensor.sum x_filled ~dim:[1] ~keepdim:true
                  |> Tensor.div (Tensor.sum mask ~dim:[1] ~keepdim:true |> Tensor.to_type Tensor.Float) in
  Tensor.where mask x_filled row_means

let bootstrap_confidence_intervals model x num_bootstrap alpha =
  let n, t = Tensor.shape2_exn x in
  let bootstrap_estimates = Parany.Cores.parany ~chunksize:1 ~core:4 ~f:(fun _ ->
    let bootstrap_indices = Tensor.randint ~high:t ~size:[t] ~dtype:(Tensor.kind x) in
    let x_bootstrap = Tensor.index_select x ~dim:1 ~index:bootstrap_indices in
    let bootstrap_model = fit model x_bootstrap 100 `PCA in
    bootstrap_model.b
  ) (List.init num_bootstrap (fun _ -> ()))
  in
  let stacked_estimates = Tensor.stack bootstrap_estimates ~dim:0 in
  let lower_percentile = (alpha /. 2.0) *. 100.0 in
  let upper_percentile = (1.0 -. alpha /. 2.0) *. 100.0 in
  let sorted_estimates = Tensor.sort stacked_estimates ~dim:0 |> fst in
  let lower_bound = Tensor.index_select sorted_estimates ~dim:0 ~index:(Tensor.of_int0 (int_of_float lower_percentile)) in
  let upper_bound = Tensor.index_select sorted_estimates ~dim:0 ~index:(Tensor.of_int0 (int_of_float upper_percentile)) in
  (lower_bound, upper_bound)

let compute_factor_contributions model x =
  let factors = estimate_factors_pca model x in
  let contributions = Tensor.abs (Tensor.mm model.b factors) in
  let total_contribution = Tensor.sum contributions ~dim:[0] ~keepdim:true in
  Tensor.div contributions total_contribution

let plot_factor_loadings model =
  let loadings = Tensor.to_float_array2 model.b in
  let plt = Plot.create "factor_loadings.png" in
  Plot.heatmap loadings plt;
  Plot.output plt

let plot_factors model x =
  let factors = estimate_factors_pca model x |> Tensor.to_float_array2 in
  let plt = Plot.create "factors.png" in
  Array.iteri (fun i factor ->
    Plot.plot ~h:plt ~spec:[ RGB (Random.int 256, Random.int 256, Random.int 256) ] (Array.to_list factor)
  ) factors;
  Plot.output plt

let diagnostic_tests model x =
  let residuals = Tensor.sub x (forward model x) in
  let acf = Stats.acf (Tensor.to_float_array residuals) 10 in
  let ljung_box = Stats.ljung_box (Tensor.to_float_array residuals) 10 in
  let normality = Stats.jarque_bera (Tensor.to_float_array residuals) in
  (acf, ljung_box, normality)

let interpolate_mixed_frequency data =
  let sorted_data = List.sort (fun a b -> compare a.date b.date) data in
  let daily_data = List.map (fun obs ->
    match obs.frequency with
    | Daily -> [obs]
    | Weekly -> List.init 7 (fun i -> {obs with date = obs.date +. float_of_int i})
    | Monthly -> List.init 30 (fun i -> {obs with date = obs.date +. float_of_int i})
    | Quarterly -> List.init 90 (fun i -> {obs with date = obs.date +. float_of_int i})
  ) sorted_data |> List.flatten in
  let interpolated = Interpolate.linear 
    (Array.of_list (List.map (fun obs -> obs.date) daily_data))
    (Array.of_list (List.map (fun obs -> obs.value) daily_data))
  in
  interpolated

let impulse_response model horizon shock =
  let irf = Array.make_matrix model.r horizon 0. in
  let shock_tensor = Tensor.of_float1 shock in
  let rec compute_irf h current_shock =
    if h >= horizon then ()
    else
      let response = match model.var_coef with
        | Some coef -> Tensor.mm coef current_shock
        | None -> current_shock
      in
      Array.blit (Tensor.to_float1 response) 0 irf h model.r;
      compute_irf (h + 1) response
  in
  compute_irf 0 shock_tensor;
  irf

let forecast_evaluation model x h =
  let n_test = min h (Tensor.size x 1 - model.t) in
  let x_test = Tensor.narrow x ~dim:1 ~start:model.t ~length:n_test in
  let forecast = forecast_var model (Tensor.narrow x ~dim:1 ~start:0 ~length:model.t) n_test in
  let mse = Tensor.mse x_test forecast in
  let mae = Tensor.mean (Tensor.abs (Tensor.sub x_test forecast)) in
  (mse, mae)

let hannan_quinn_criterion model x =
  let n = Float.of_int model.n in
  let t = Float.of_int model.t in
  let r = Float.of_int model.r in
  let residuals = Tensor.sub x (forward model x) in
  let mse = Tensor.mean (Tensor.pow residuals ~exponent:2.) in
  Tensor.log mse +. (2. *. r *. log (log t)) /. t

let select_optimal_factors_hq x max_factors =
  let n, t = Tensor.shape2_exn x in
  let criterion_values = Parany.Cores.parany ~chunksize:1 ~core:4 ~f:(fun r ->
    let model = create n (r + 1) t Static in
    let fitted_model = fit model x 100 `PCA in
    let hq = hannan_quinn_criterion fitted_model x in
    (r + 1, hq)
  ) (List.init max_factors (fun r -> r))
  in
  List.fold_left (fun (best_r, best_hq) (r, hq) ->
    if hq < best_hq then (r, hq) else (best_r, best_hq)
  ) (0, Float.infinity) criterion_values |> fst

let detect_structural_breaks model x window_size =
  let n, t = Tensor.shape2_exn x in
  let cusum = Tensor.zeros [n; t] in
  let residuals = Tensor.sub x (forward model x) in
  let cumsum_residuals = Tensor.cumsum residuals ~dim:1 in
  let std_dev = Tensor.std residuals ~dim:[1] ~unbiased:true ~keepdim:true in
  for i = window_size to t - 1 do
    let window_cumsum = Tensor.sub 
      (Tensor.select cumsum_residuals ~dim:1 ~index:i)
      (Tensor.select cumsum_residuals ~dim:1 ~index:(i - window_size))
    in
    let _ = Tensor.div_inplace window_cumsum std_dev in
    let _ = Tensor.set cusum ~index:i window_cumsum in
    ()
  done;
  cusum

let regularized_fit model x max_iter lambda =
  let rec fit_iter model iter =
    if iter >= max_iter then model
    else
      let factors = estimate_factors_pca model x in
      let loadings = estimate_loadings_ols model x factors in
      let regularized_loadings = Tensor.sub loadings (Tensor.mul loadings lambda) in
      let new_model = { model with b = regularized_loadings; f = factors } in
      fit_iter new_model (iter + 1)
  in
  fit_iter model 0

let summary_statistics model x =
  let factors = estimate_factors_pca model x in
  let loadings = model.b in
  let factor_mean = Tensor.mean factors ~dim:[1] in
  let factor_std = Tensor.std factors ~dim:[1] ~unbiased:true in
  let loading_mean = Tensor.mean loadings ~dim:[0] in
  let loading_std = Tensor.std loadings ~dim:[0] ~unbiased:true in
  let explained_variance = compute_factor_contributions model x in
  (factor_mean, factor_std, loading_mean, loading_std, explained_variance)

let plot_factor_heatmap model x =
  let factors = estimate_factors_pca model x |> Tensor.to_float_array2 in
  let plt = Plot.create "factor_heatmap.png" in
  Plot.heatmap factors plt;
  Plot.output plt