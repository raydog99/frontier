open Torch

type t = {
  returns: float array;
  risk_free_rate: float;
  max_lag: int;
}

exception InsufficientData of string
exception InvalidParameter of string

let create returns risk_free_rate max_lag =
  if Array.length returns < 2 then
    raise (InsufficientData "At least 2 return observations are required")
  else if max_lag < 1 then
    raise (InvalidParameter "max_lag must be at least 1")
  else
    { returns; risk_free_rate; max_lag = min max_lag (Array.length returns / 2) }

let calculate_mean_and_std returns =
  let returns_tensor = Tensor.(of_float_array returns |> reshape ~shape:[1; -1]) in
  let mean = Tensor.mean returns_tensor ~dim:[1] ~keepdim:true in
  let std = Tensor.std returns_tensor ~dim:[1] ~unbiased:true ~keepdim:true in
  (Tensor.to_float0_exn mean, Tensor.to_float0_exn std)

let safe_div a b =
  if Float.abs b < Float.epsilon then 0. else a /. b

let calculate_iid_sharpe_ratio t =
  let mean, std = calculate_mean_and_std t.returns in
  safe_div (mean -. t.risk_free_rate) std

let calculate_iid_sharpe_ratio_standard_error t =
  let sr = calculate_iid_sharpe_ratio t in
  let n = Array.length t.returns in
  sqrt ((1. +. 0.5 *. sr ** 2.) /. float_of_int n)

let calculate_autocorrelation returns lag =
  let returns_tensor = Tensor.(of_float_array returns |> reshape ~shape:[1; -1]) in
  let n = Tensor.shape returns_tensor |> List.hd in
  let shifted = Tensor.narrow returns_tensor ~dim:1 ~start:lag ~length:(n - lag) in
  let original = Tensor.narrow returns_tensor ~dim:1 ~start:0 ~length:(n - lag) in
  let cov = Tensor.mean (Tensor.mul shifted original) in
  let var = Tensor.var returns_tensor ~unbiased:false ~dim:[1] in
  safe_div (Tensor.to_float0_exn cov) (Tensor.to_float0_exn var)

let calculate_gmm_sharpe_ratio t =
  let mean, std = calculate_mean_and_std t.returns in
  let excess_return = mean -. t.risk_free_rate in
  
  let autocorrelations = Array.init t.max_lag (fun lag -> calculate_autocorrelation t.returns (lag + 1)) in
  let adjustment_factor = 1. +. 2. *. (Array.fold_left (+.) 0. autocorrelations) in
  
  safe_div excess_return (std *. sqrt adjustment_factor)

let calculate_gmm_sharpe_ratio_standard_error t =
  let n = Array.length t.returns in
  let gmm_sr = calculate_gmm_sharpe_ratio t in
  sqrt ((1. +. 0.5 *. gmm_sr ** 2.) /. float_of_int n)

let time_aggregate_sharpe_ratio t q =
  if q < 1 then raise (InvalidParameter "q must be at least 1");
  let sr = calculate_gmm_sharpe_ratio t in
  let autocorrelations = Array.init (min (q - 1) t.max_lag) (fun lag -> calculate_autocorrelation t.returns (lag + 1)) in
  let adjustment_factor = 1. +. 2. *. (Array.fold_left (+.) 0. (Array.mapi (fun i ac -> float_of_int (q - i - 1) *. ac) autocorrelations)) in
  sr *. sqrt (float_of_int q /. adjustment_factor)

let time_aggregate_sharpe_ratio_standard_error t q =
  if q < 1 then raise (InvalidParameter "q must be at least 1");
  let n = Array.length t.returns in
  let agg_sr = time_aggregate_sharpe_ratio t q in
  sqrt ((1. +. 0.5 *. agg_sr ** 2.) /. float_of_int n)

let ljung_box_test t =
  let n = Array.length t.returns in
  let autocorrelations = Array.init t.max_lag (fun lag -> calculate_autocorrelation t.returns (lag + 1)) in
  let q_stat = Array.fold_left (fun acc rho -> 
    acc +. (rho ** 2. /. float_of_int (n - lag - 1))
  ) 0. autocorrelations in
  q_stat *. float_of_int n *. (float_of_int n + 2.)

let jarque_bera_test t =
  let returns_tensor = Tensor.(of_float_array t.returns |> reshape ~shape:[1; -1]) in
  let n = float_of_int (Array.length t.returns) in
  let mean, std = calculate_mean_and_std t.returns in
  let skewness = Tensor.(sum (pow (sub returns_tensor (float_tensor mean)) 3) |> to_float0_exn) in
  let skewness = skewness /. (n *. std ** 3.) in
  let kurtosis = Tensor.(sum (pow (sub returns_tensor (float_tensor mean)) 4) |> to_float0_exn) in
  let kurtosis = kurtosis /. (n *. std ** 4.) -. 3. in
  n /. 6. *. (skewness ** 2. +. 0.25 *. kurtosis ** 2.)

let confidence_interval t confidence_level =
  if confidence_level <= 0. || confidence_level >= 1. then
    raise (InvalidParameter "Confidence level must be between 0 and 1");
  let sr = calculate_gmm_sharpe_ratio t in
  let se = calculate_gmm_sharpe_ratio_standard_error t in
  let z_score = 1.96 (* Approximation for 95% CI *) in
  (sr -. z_score *. se, sr +. z_score *. se)

let compare_sharpe_ratios t1 t2 =
  let sr1 = calculate_gmm_sharpe_ratio t1 in
  let sr2 = calculate_gmm_sharpe_ratio t2 in
  let se1 = calculate_gmm_sharpe_ratio_standard_error t1 in
  let se2 = calculate_gmm_sharpe_ratio_standard_error t2 in
  let z_stat = (sr1 -. sr2) /. sqrt (se1 ** 2. +. se2 ** 2.) in
  (* Approximation of p-value calculation *)
  2. *. (1. -. (0.5 *. (1. +. erf (abs_float z_stat /. sqrt 2.))))

let rolling_sharpe_ratio t window_size =
  if window_size < 2 || window_size > Array.length t.returns then
    raise (InvalidParameter "Invalid window size");
  Array.init (Array.length t.returns - window_size + 1) (fun i ->
    let window = Array.sub t.returns i window_size in
    let window_t = create window t.risk_free_rate t.max_lag in
    calculate_gmm_sharpe_ratio window_t
  )

let bootstrap_sharpe_ratio t num_samples =
  if num_samples < 1 then raise (InvalidParameter "num_samples must be at least 1");
  let n = Array.length t.returns in
  Array.init num_samples (fun _ ->
    let sampled_returns = Array.init n (fun _ -> t.returns.(Random.int n)) in
    let sampled_t = create sampled_returns t.risk_free_rate t.max_lag in
    calculate_gmm_sharpe_ratio sampled_t
  )

let adjust_for_autocorrelation t =
  let adjusted_returns = Array.copy t.returns in
  let mean = Array.fold_left (+.) 0. adjusted_returns /. float_of_int (Array.length adjusted_returns) in
  for i = 1 to Array.length adjusted_returns - 1 do
    let ac = calculate_autocorrelation adjusted_returns 1 in
    adjusted_returns.(i) <- adjusted_returns.(i) -. ac *. (adjusted_returns.(i-1) -. mean)
    done;
  create adjusted_returns t.risk_free_rate t.max_lag

let detect_outliers t threshold =
  let mean, std = calculate_mean_and_std t.returns in
  Array.to_list t.returns
  |> List.mapi (fun i x -> (i, abs_float ((x -. mean) /. std)))
  |> List.filter (fun (_, z_score) -> z_score > threshold)
  |> List.map fst
  |> Array.of_list

let winsorize t percentile =
  if percentile <= 0. || percentile >= 0.5 then raise (InvalidParameter "Percentile must be between 0 and 0.5");
  let sorted_returns = Array.copy t.returns in
  Array.sort compare sorted_returns;
  let n = Array.length sorted_returns in
  let lower_bound = sorted_returns.(int_of_float (float_of_int n *. percentile)) in
  let upper_bound = sorted_returns.(int_of_float (float_of_int n *. (1. -. percentile))) in
  let winsorized_returns = Array.map (fun x -> max lower_bound (min x upper_bound)) t.returns in
  create winsorized_returns t.risk_free_rate t.max_lag

let generate_summary_report t =
  let iid_sr = calculate_iid_sharpe_ratio t in
  let iid_se = calculate_iid_sharpe_ratio_standard_error t in
  let gmm_sr = calculate_gmm_sharpe_ratio t in
  let gmm_se = calculate_gmm_sharpe_ratio_standard_error t in
  let annual_sr = time_aggregate_sharpe_ratio t 12 in
  let annual_se = time_aggregate_sharpe_ratio_standard_error t 12 in
  let lb_stat = ljung_box_test t in
  let jb_stat = jarque_bera_test t in
  Printf.sprintf
    "Sharpe Ratio Analysis Summary\n\
     -------------------------------\n\
     Number of observations: %d\n\
     Risk-free rate: %.4f\n\n\
     IID Sharpe Ratio: %.4f (SE: %.4f)\n\
     GMM Sharpe Ratio: %.4f (SE: %.4f)\n\
     Annual Sharpe Ratio: %.4f (SE: %.4f)\n\n\
     Ljung-Box test statistic: %.4f\n\
     Jarque-Bera test statistic: %.4f"
    (Array.length t.returns)
    t.risk_free_rate
    iid_sr iid_se
    gmm_sr gmm_se
    annual_sr annual_se
    lb_stat
    jb_stat

let parallel_bootstrap_sharpe_ratio t num_samples num_threads =
  if num_samples < 1 || num_threads < 1 then
    raise (InvalidParameter "num_samples and num_threads must be at least 1");
  let samples_per_thread = num_samples / num_threads in
  let remainder = num_samples mod num_threads in
  let results = Array.make num_samples 0. in
  let worker thread_id =
    let start = thread_id * samples_per_thread + min thread_id remainder in
    let end_ = start + samples_per_thread + (if thread_id < remainder then 1 else 0) in
    for i = start to end_ - 1 do
      let sampled_returns = Array.init (Array.length t.returns) (fun _ -> t.returns.(Random.int (Array.length t.returns))) in
      let sampled_t = create sampled_returns t.risk_free_rate t.max_lag in
      results.(i) <- calculate_gmm_sharpe_ratio sampled_t
    done
  in
  let threads = Array.init num_threads (fun i -> Thread.create worker i) in
  Array.iter Thread.join threads;
  results

let maximum_drawdown t =
  let cumulative_returns = Array.make (Array.length t.returns) 1. in
  for i = 1 to Array.length t.returns - 1 do
    cumulative_returns.(i) <- cumulative_returns.(i-1) *. (1. +. t.returns.(i))
  done;
  let mutable max_so_far = cumulative_returns.(0) in
  let mutable max_drawdown = 0. in
  Array.iter (fun cr ->
    max_so_far <- max max_so_far cr;
    max_drawdown <- max max_drawdown ((max_so_far -. cr) /. max_so_far)
  ) cumulative_returns;
  max_drawdown

let sortino_ratio t =
  let mean, _ = calculate_mean_and_std t.returns in
  let excess_return = mean -. t.risk_free_rate in
  let downside_returns = Array.map (fun r -> min (r -. t.risk_free_rate) 0.) t.returns in
  let downside_deviation = 
    let sum_squared = Array.fold_left (fun acc r -> acc +. r *. r) 0. downside_returns in
    sqrt (sum_squared /. float_of_int (Array.length downside_returns))
  in
  safe_div excess_return downside_deviation

let calmar_ratio t =
  let mean, _ = calculate_mean_and_std t.returns in
  let excess_return = mean -. t.risk_free_rate in
  let max_dd = maximum_drawdown t in
  safe_div excess_return max_dd

let omega_ratio t ?(threshold=None) () =
  let threshold = match threshold with
    | Some t -> t
    | None -> t.risk_free_rate
  in
  let gains = ref 0. in
  let losses = ref 0. in
  Array.iter (fun r ->
    if r > threshold then gains := !gains +. (r -. threshold)
    else losses := !losses +. (threshold -. r)
  ) t.returns;
  safe_div !gains !losses