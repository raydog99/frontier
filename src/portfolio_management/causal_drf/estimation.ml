open Torch

type ckte_estimate = {
  point_estimate: Tensor.t;
  standard_error: Tensor.t;
  kernel_type: kernel_type;
  effective_sample_size: float;
}

type confidence_band = {
  lower: Tensor.t;
  upper: Tensor.t;
  pointwise_coverage: float;
  uniform_coverage: float;
  band_width: float;
}

let estimate_ckte data forest x =
  let n = Tensor.size2 data.features 0 in
  let weights = get_weights forest x n in
  
  (* Calculate treatment and control group statistics *)
  let treat_indices, control_indices = Array.init n (fun i -> i)
    |> Array.partition (fun i -> 
      Tensor.get_float1 data.treatment i > 0.5) in
  
  (* Calculate effective sample sizes *)
  let n_treat_eff = Array.fold_left (fun acc i -> 
    acc +. weights.(i)) 0.0 treat_indices in
  let n_control_eff = Array.fold_left (fun acc i -> 
    acc +. weights.(i)) 0.0 control_indices in
  
  (* Compute weighted outcomes *)
  let treat_outcome = ref (Tensor.zeros_like (Tensor.get data.outcome 0)) in
  let control_outcome = ref (Tensor.zeros_like (Tensor.get data.outcome 0)) in
  
  Array.iter (fun i ->
    let w = weights.(i) in
    let y = Tensor.get data.outcome i in
    if Tensor.get_float1 data.treatment i > 0.5 then
      treat_outcome := Tensor.add !treat_outcome 
        (Tensor.mul_scalar y (w /. n_treat_eff))
    else
      control_outcome := Tensor.add !control_outcome 
        (Tensor.mul_scalar y (w /. n_control_eff))
  ) (Array.init n (fun i -> i));
  
  (* Calculate standard errors *)
  let calc_group_variance indices mean =
    let variance = ref (Tensor.zeros_like mean) in
    Array.iter (fun i ->
      let w = weights.(i) in
      let y = Tensor.get data.outcome i in
      let diff = Tensor.sub y mean in
      variance := Tensor.add !variance 
        (Tensor.mul_scalar (Tensor.mul diff diff) (w *. w))
    ) indices;
    !variance
  in
  
  let treat_var = calc_group_variance treat_indices !treat_outcome in
  let control_var = calc_group_variance control_indices !control_outcome in
  let std_error = Tensor.sqrt (Tensor.add treat_var control_var) in
  
  {
    point_estimate = Tensor.sub !treat_outcome !control_outcome;
    standard_error = std_error;
    kernel_type = Gaussian 1.0;
    effective_sample_size = min n_treat_eff n_control_eff;
  }

let construct_uniform_bands data forest x alpha =
  let n = Tensor.size2 data.features 0 in
  let n_bootstrap = 1000 in
  
  (* Generate bootstrap samples *)
  let bootstrap_estimates = Array.init n_bootstrap (fun _ ->
    let bootstrap_indices = Array.init n (fun _ -> Random.int n) in
    let bootstrap_data = {
      features = Tensor.index_select data.features 0 
        (Tensor.of_int1 bootstrap_indices);
      treatment = Tensor.index_select data.treatment 0 
        (Tensor.of_int1 bootstrap_indices);
      outcome = Tensor.index_select data.outcome 0 
        (Tensor.of_int1 bootstrap_indices)
    } in
    estimate_ckte bootstrap_data forest x
  ) in
  
  let calculate_critical_value estimates alpha =
    let standardized_sups = Array.map (fun est ->
      let standardized = Tensor.div 
        (Tensor.sub est.point_estimate estimates.(0).point_estimate)
        est.standard_error in
      Tensor.max standardized |> Tensor.float_value
    ) estimates in
    Array.sort compare standardized_sups;
    standardized_sups.(int_of_float ((1.0 -. alpha) *. float_of_int n_bootstrap))
  in
  
  let c_alpha = calc_critical_value bootstrap_estimates alpha in
  let estimate = estimate_ckte data forest x in
  
  let lower = Tensor.sub estimate.point_estimate 
    (Tensor.mul_scalar estimate.standard_error c_alpha) in
  let upper = Tensor.add estimate.point_estimate 
    (Tensor.mul_scalar estimate.standard_error c_alpha) in
  
  let calculate_converge bands estimates =
    let count = ref 0 in
    Array.iter (fun est ->
      let covered = Tensor.all (Tensor.logical_and
        (Tensor.ge est.point_estimate bands.lower)
        (Tensor.le est.point_estimate bands.upper)) in
      if covered then incr count
    ) estimates;
    float_of_int !count /. float_of_int (Array.length estimates)
  in
  
  let bands = {
    lower;
    upper;
    pointwise_coverage = calc_coverage 
      {lower; upper; pointwise_coverage=0.; uniform_coverage=0.; band_width=0.} 
      bootstrap_estimates;
    uniform_coverage = calc_coverage 
      {lower; upper; pointwise_coverage=0.; uniform_coverage=0.; band_width=0.} 
      bootstrap_estimates;
    band_width = Tensor.mean (Tensor.sub upper lower) |> Tensor.float_value;
  } in
  bands

let calculate_witness_function data kernel x y =
  let ckte = estimate_ckte data 
    (build_forest data 100 10) x in
  let k_y = evaluate_kernel kernel y in
  Tensor.dot ckte.point_estimate k_y |> Tensor.float_value

let estimate_ckte_with_samples data forest x half_samples =
  let n = Tensor.size2 data.features 0 in
  let predictions = Array.map (fun sample ->
    let subset_data = {
      features = Tensor.index_select data.features 0 (Tensor.of_int1 sample.indices);
      treatment = Tensor.index_select data.treatment 0 (Tensor.of_int1 sample.indices);
      outcome = Tensor.index_select data.outcome 0 (Tensor.of_int1 sample.indices)
    } in
    let weights = get_weights forest x (Array.length sample.indices) in
    calculate_weighted_prediction subset_data weights
  ) half_samples in
  
  let point_estimate = Array.fold_left (fun acc pred ->
    Tensor.add acc pred
  ) (Tensor.zeros_like predictions.(0)) predictions
  |> fun t -> Tensor.div t (float_of_int (Array.length half_samples)) in
  
  let std_error = calculate_standard_error data predictions half_samples in
  
  {
    point_estimate;
    standard_error = std_error;
    kernel_type = Gaussian 1.0;
    effective_sample_size = float_of_int (Array.length half_samples)
  }

let get_critical_value alpha =
  let df = 100.0 in
  Statistics.t_quantile (1.0 -. alpha /. 2.0) df

let calculate_weighted_prediction data weights =
  let n = Array.length weights in
  let prediction = ref (Tensor.zeros_like (Tensor.get data.outcome 0)) in
  for i = 0 to n-1 do
    prediction := Tensor.add !prediction
      (Tensor.mul_scalar (Tensor.get data.outcome i) weights.(i))
  done;
  !prediction

let calculate_standard_error data predictions half_samples =
  let n_samples = Array.length half_samples in
  let n_obs = Array.length predictions in
  
  (* Calculate mean prediction *)
  let mean_pred = Array.fold_left (fun acc pred ->
    Tensor.add acc pred
  ) (Tensor.zeros_like predictions.(0)) predictions
  |> fun t -> Tensor.div t (float_of_int n_obs) in
  
  (* Calculate variance components *)
  let between_var = Array.fold_left (fun acc half_sample ->
    let sample_pred = Array.fold_left2 (fun acc w p ->
      Tensor.add acc (Tensor.mul_scalar p w)
    ) (Tensor.zeros_like predictions.(0)) half_sample.weights predictions in
    let diff = Tensor.sub sample_pred mean_pred in
    Tensor.add acc (Tensor.mul diff diff)
  ) (Tensor.zeros_like mean_pred) half_samples
  |> fun t -> Tensor.div t (float_of_int (n_samples - 1)) in
  
  let within_var = Array.fold_left (fun acc half_sample ->
    let group_var = calculate_group_variance data [|half_sample|] predictions in
    Tensor.add acc group_var.(0)
  ) (Tensor.zeros_like mean_pred) half_samples
  |> fun t -> Tensor.div t (float_of_int n_samples) in
  
  Tensor.sqrt (Tensor.add between_var within_var)

let verify_coverage confidence_band true_effect grid_points =
  let n_points = Array.length grid_points in
  let covered = ref 0 in
  
  Array.iter (fun y ->
    let true_val = Tensor.get true_effect y in
    let lower = Tensor.get confidence_band.lower y in
    let upper = Tensor.get confidence_band.upper y in
    if true_val >= lower && true_val <= upper then
      incr covered
  ) grid_points;
  
  float_of_int !covered /. float_of_int n_points >= 0.95