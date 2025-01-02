open Torch

type frequency = Daily | Weekly | Monthly | Quarterly

type observation = {
  value : float;
  frequency : frequency;
  date : float;
}

type model_type =
  | Unrestricted
  | Restricted of int
  | FAVAR of int (* p: VAR order *)

type t = {
  n : int;
  q : int;
  r : int;
  t : int;
  model_type : model_type;
  x : Tensor.t;
  common : Tensor.t;
  idiosyncratic : Tensor.t;
  b : Tensor.t option;
  f : Tensor.t option;
  var_coef : Tensor.t option; (* For FAVAR *)
}

let create n q t x model_type =
  let r = match model_type with
    | Unrestricted -> q
    | Restricted s -> q * (s + 1)
    | FAVAR _ -> q
  in
  let common = Tensor.zeros [n; t] in
  let idiosyncratic = Tensor.zeros [n; t] in
  { n; q; r; t; model_type; x; common; idiosyncratic; b = None; f = None; var_coef = None }

let handle_missing_data x =
  let mask = Tensor.(x.isnan () |> logical_not) in
  let x_filled = Tensor.where mask x (Tensor.zeros_like x) in
  let row_means = Tensor.(sum x_filled ~dim:[1] ~keepdim:true / 
                          (sum mask ~dim:[1] ~keepdim:true |> to_type Float)) in
  Tensor.where mask x_filled row_means

let estimate_spectral_density x num_lags =
  let n, t = Tensor.shape2_exn x in
  let x_centered = Tensor.(x - mean x ~dim:[1] ~keepdim:true) in
  let fft = Tensor.fft x_centered ~signal_ndim:1 ~normalized:true in
  let spec = Tensor.(fft * conj fft) in
  Tensor.(spec / of_float (Float.of_int t))

let dynamic_pca x q num_freqs =
  let n, t = Tensor.shape2_exn x in
  let spec = estimate_spectral_density x num_freqs in
  let eigvals, eigvecs = Tensor.symeig spec ~eigenvectors:true in
  Tensor.narrow eigvecs ~dim:1 ~start:0 ~length:q

let fit model lambda =
  match model.model_type with
  | Unrestricted ->
      let factors = dynamic_pca model.x model.q (min model.t 20) in
      let common = Tensor.(mm factors (transpose factors ~dim0:0 ~dim1:1)) in
      let idiosyncratic = Tensor.(model.x - common) in
      { model with common; idiosyncratic; f = Some factors }
  | Restricted s ->
      let factors = dynamic_pca model.x model.q (min model.t 20) in
      let f = Tensor.cat (List.init (s + 1) (fun i -> 
        Tensor.narrow factors ~dim:1 ~start:i ~length:(model.t - s)
      )) ~dim:0 in
      let x_lagged = Tensor.narrow model.x ~dim:1 ~start:s ~length:(model.t - s) in
      let b = Tensor.(mm x_lagged (pinverse f)) in
      let common = Tensor.(mm b f) in
      let idiosyncratic = Tensor.(x_lagged - common) in
      { model with common; idiosyncratic; b = Some b; f = Some f }
  | FAVAR p ->
      let factors = dynamic_pca model.x model.q (min model.t 20) in
      let x_augmented = Tensor.cat [factors; model.x] ~dim:0 in
      let var_coef = estimate_var_coefficients x_augmented p in
      let common = Tensor.(mm factors (transpose factors ~dim0:0 ~dim1:1)) in
      let idiosyncratic = Tensor.(model.x - common) in
      { model with common; idiosyncratic; f = Some factors; var_coef = Some var_coef }

and estimate_var_coefficients factors p =
  let t = Tensor.size factors 1 in
  let y = Tensor.narrow factors ~dim:1 ~start:p ~length:(t - p) in
  let x = Tensor.cat (List.init p (fun i ->
    Tensor.narrow factors ~dim:1 ~start:(p - i - 1) ~length:(t - p)
  )) ~dim:0 in
  let x_t = Tensor.transpose x ~dim0:0 ~dim1:1 in
  let inv_xx = Tensor.pinverse Tensor.(mm x_t x) in
  Tensor.(mm (mm inv_xx x_t) y)

let forecast model h =
  match model.model_type with
  | Unrestricted | Restricted _ ->
      let factors = Option.get model.f in
      let var_coef = estimate_var_coefficients factors 1 in
      let last_factor = Tensor.select factors ~dim:1 ~index:(Tensor.size factors 1 - 1) in
      let rec forecast_iter current_factor h_left acc =
        if h_left = 0 then List.rev acc
        else
          let next_factor = Tensor.(mm var_coef (unsqueeze current_factor ~dim:1)) in
          forecast_iter (Tensor.squeeze_dim next_factor ~dim:1) (h_left - 1) (next_factor :: acc)
      in
      let future_factors = forecast_iter last_factor h [] |> Tensor.cat ~dim:1 in
      Tensor.(mm (Option.get model.b) future_factors)
  | FAVAR _ ->
      let f = Option.get model.f in
      let var_coef = Option.get model.var_coef in
      let x_augmented = Tensor.cat [f; model.x] ~dim:0 in
      let last_obs = Tensor.select x_augmented ~dim:1 ~index:(Tensor.size x_augmented 1 - 1) in
      let rec forecast_iter current_obs h_left acc =
        if h_left = 0 then List.rev acc
        else
          let next_obs = Tensor.(mm var_coef (unsqueeze current_obs ~dim:1)) in
          forecast_iter (Tensor.squeeze_dim next_obs ~dim:1) (h_left - 1) (next_obs :: acc)
      in
      let future_obs = forecast_iter last_obs h [] |> Tensor.cat ~dim:1 in
      Tensor.narrow future_obs ~dim:0 ~start:0 ~length:model.n

let compute_explained_variance model =
  let total_var = Tensor.var model.x ~dim:[1] ~unbiased:true ~keepdim:true in
  let common_var = Tensor.var model.common ~dim:[1] ~unbiased:true ~keepdim:true in
  Tensor.(mean (common_var / total_var)) |> Tensor.to_float0_exn

let information_criterion model max_factors =
  let n, t = Tensor.shape2_exn model.x in
  List.init max_factors (fun k ->
    let temp_model = create model.n (k + 1) model.t model.x model.model_type |> fun m -> fit m 0. in
    let residuals = Tensor.(model.x - temp_model.common) in
    let sigma2 = Tensor.(mean (pow residuals ~exponent:2.)) |> Tensor.to_float0_exn in
    let penalty = (float n +. float t) /. (float n *. float t) *. (float k) *. (log (min n t |> float) /. (float (min n t))) in
    sigma2 +. penalty
  )

let select_number_of_factors model max_factors =
  let criteria = information_criterion model max_factors in
  List.mapi (fun i c -> (i + 1, c)) criteria
  |> List.fold_left (fun (best_k, best_ic) (k, ic) ->
       if ic < best_ic then (k, ic) else (best_k, best_ic)
     ) (0, Float.infinity)
  |> fst

let bootstrap_confidence_intervals model num_bootstrap alpha =
  let n, t = Tensor.shape2_exn model.x in
  let bootstrap_estimates = List.init num_bootstrap (fun _ ->
    let bootstrap_indices = Tensor.(randint ~high:t ~size:[t] (Kind.T (kind model.x))) in
    let x_bootstrap = Tensor.index_select model.x ~dim:1 ~index:bootstrap_indices in
    let bootstrap_model = fit {model with x = x_bootstrap} 0. in
    Option.get bootstrap_model.b
  ) in
  let stacked_estimates = Tensor.stack bootstrap_estimates ~dim:0 in
  let lower_percentile = (alpha /. 2.0) *. 100.0 in
  let upper_percentile = (1.0 -. alpha /. 2.0) *. 100.0 in
  let sorted_estimates = Tensor.sort stacked_estimates ~dim:0 |> fst in
  let lower_bound = Tensor.index_select sorted_estimates ~dim:0 ~index:(Tensor.of_int0 (int_of_float lower_percentile)) in
  let upper_bound = Tensor.index_select sorted_estimates ~dim:0 ~index:(Tensor.of_int0 (int_of_float upper_percentile)) in
  (lower_bound, upper_bound)

let detect_structural_breaks model window_size =
  let n, t = Tensor.shape2_exn model.x in
  let cusum = Tensor.zeros [n; t] in
  let residuals = Tensor.(model.x - model.common) in
  let cumsum_residuals = Tensor.cumsum residuals ~dim:1 in
  let std_dev = Tensor.std residuals ~dim:[1] ~unbiased:true ~keepdim:true in
  for i = window_size to t - 1 do
    let window_cumsum = Tensor.(
      select cumsum_residuals ~dim:1 ~index:i - 
      select cumsum_residuals ~dim:1 ~index:(i - window_size)
    ) in
    let _ = Tensor.(window_cumsum /= std_dev) in
    let _ = Tensor.set cusum ~index:i window_cumsum in
    ()
  done;
  cusum

let summary_statistics model =
  let factors = Option.get model.f in
  let factor_mean = Tensor.mean factors ~dim:[1] in
  let factor_std = Tensor.std factors ~dim:[1] ~unbiased:true in
  let loadings = Option.get model.b in
  let loading_mean = Tensor.mean loadings ~dim:[0] in
  let loading_std = Tensor.std loadings ~dim:[0] ~unbiased:true in
  (factor_mean, factor_std, loading_mean, loading_std)