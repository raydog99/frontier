open Torch
open Types
open Error_handling
open Optimization

let estimate_variance_cc prices =
  let n = Tensor.shape prices |> List.hd in
  let returns = Tensor.(log (slice prices ~dim:0 ~start:1 ~end:n) - log (slice prices ~dim:0 ~start:0 ~end:(n-1))) in
  let mean_return = Tensor.mean returns in
  let squared_deviations = Tensor.((returns - mean_return) * (returns - mean_return)) in
  Tensor.mean squared_deviations

let estimate_variance_p highs lows =
  let n = Tensor.shape highs |> List.hd in
  let range_squared = Tensor.((log highs - log lows) ** (Scalar.f 2.0)) in
  Tensor.(mean range_squared / (Scalar.f 4.0 * Scalar.f (log 2.0)))

let estimate_variance_rs highs lows closes =
  let n = Tensor.shape highs |> List.hd in
  let u = Tensor.(log highs - log closes) in
  let d = Tensor.(log lows - closes) in
  Tensor.mean Tensor.((u * (u - closes) + d * (d - closes)))

let estimate_variance_o opens =
  let n = Tensor.shape opens |> List.hd in
  let o = Tensor.(log opens) in
  let mean_o = Tensor.mean o in
  let squared_deviations = Tensor.((o - mean_o) * (o - mean_o)) in
  Tensor.(sum squared_deviations / (Scalar.f (float_of_int (n - 1))))

let estimate_variance_c closes =
  let n = Tensor.shape closes |> List.hd in
  let c = Tensor.(log closes) in
  let mean_c = Tensor.mean c in
  let squared_deviations = Tensor.((c - mean_c) * (c - mean_c)) in
  Tensor.(sum squared_deviations / (Scalar.f (float_of_int (n - 1))))

let calculate_k0 n alpha =
  let n_float = float_of_int n in
  (alpha -. 1.0) /. (alpha +. (n_float +. 1.0) /. (n_float -. 1.0))

let estimate_variance ?(alpha = 1.34) opens highs lows closes =
  let n = Tensor.shape opens |> List.hd in
  let v_o = estimate_variance_o opens in
  let v_c = estimate_variance_c closes in
  let v_rs = estimate_variance_rs highs lows closes in
  let k0 = calculate_k0 n alpha in
  Tensor.(v_o + (Scalar.f k0 * v_c) + (Scalar.f (1.0 -. k0) * v_rs))

let estimate_garch_parameters_generic model returns ~max_iter ~learning_rate =
  validate_input returns;
  let n = Tensor.shape returns |> List.hd in
  let omega = Tensor.rand [] ~dtype:Tensor.Float in
  let alpha = Tensor.rand [] ~dtype:Tensor.Float in
  let gamma = Tensor.rand [] ~dtype:Tensor.Float in
  let beta = Tensor.rand [] ~dtype:Tensor.Float in
  
  let loss_fn params =
    let omega, alpha, gamma, beta = params in
    let variance = Tensor.zeros [n] ~dtype:Tensor.Float in
    Tensor.set variance [0] (Tensor.to_float0_exn omega);
    
    for t = 1 to n - 1 do
      let prev_var = Tensor.get variance [t-1] in
      let prev_return = Tensor.get returns [t-1] in
      let new_var = match model with
        | GARCH -> Tensor.(omega + alpha * prev_return * prev_return + beta * prev_var)
        | EGARCH -> 
            let log_var = Tensor.log prev_var in
            Tensor.(exp (omega + 
                         alpha * (abs prev_return / sqrt prev_var - Scalar.f (sqrt (2.0 /. Float.pi))) +
                         gamma * prev_return / sqrt prev_var +
                         beta * log_var))
        | GJR_GARCH ->
            let indicator = if Tensor.to_float0_exn prev_return < 0.0 then 1.0 else 0.0 in
            Tensor.(omega + 
                    alpha * prev_return * prev_return +
                    Scalar.f indicator * gamma * prev_return * prev_return +
                    beta * prev_var)
      in
      Tensor.set variance [t] new_var;
    done;
    
    Tensor.(mean (log variance + (returns * returns) / variance))
  in
  
  let optimized_params = adam [omega; alpha; gamma; beta] 
    ~loss_fn 
    ~learning_rate 
    ~beta1:0.9 
    ~beta2:0.999 
    ~epsilon:1e-8 
    ~max_iter 
  in
  
  match optimized_params with
  | [omega; alpha; gamma; beta] -> 
      (Tensor.to_float0_exn omega, Tensor.to_float0_exn alpha, Tensor.to_float0_exn gamma, Tensor.to_float0_exn beta)
  | _ -> failwith "Unexpected number of parameters"

let forecast_volatility model params returns horizon =
  let omega, alpha, gamma, beta = params in
  let n = Tensor.shape returns |> List.hd in
  let forecasts = Tensor.zeros [horizon] in
  let last_var = Tensor.get returns [n-1] ** (Scalar.f 2.0) in
  
  for h = 0 to horizon - 1 do
    let forecast = match model with
      | GARCH -> omega +. alpha *. last_var +. beta *. (if h = 0 then last_var else Tensor.get forecasts [h-1])
      | EGARCH ->
          let log_var = log last_var in
          exp (omega +. alpha *. (abs_float (Tensor.get returns [n-1]) /. sqrt last_var -. sqrt (2.0 /. Float.pi)) +.
               gamma *. (Tensor.get returns [n-1] /. sqrt last_var) +.
               beta *. log_var)
      | GJR_GARCH ->
          let indicator = if Tensor.get returns [n-1] < 0.0 then 1.0 else 0.0 in
          omega +. (alpha +. gamma *. indicator) *. last_var +. beta *. (if h = 0 then last_var else Tensor.get forecasts [h-1])
    in
    Tensor.set forecasts [h] forecast;
  done;
  
  forecasts

let calculate_confidence_interval forecasts confidence_level =
  let mean = Tensor.mean forecasts in
  let std = Tensor.std forecasts ~unbiased:true in
  let z_score = Statistics.normal_ppf ((1.0 +. confidence_level) /. 2.0) in
  let lower = Tensor.(mean - Scalar.f z_score * std) in
  let upper = Tensor.(mean + Scalar.f z_score * std) in
  (lower, upper)

let calculate_var returns confidence_level =
  let sorted_returns = Tensor.sort returns ~descending:false in
  let index = int_of_float (float_of_int (Tensor.shape returns |> List.hd) *. (1.0 -. confidence_level)) in
  Tensor.get sorted_returns [index]

let calculate_es returns confidence_level =
  let var = calculate_var returns confidence_level in
  let tail_returns = Tensor.masked_select returns Tensor.(returns <= var) in
  Tensor.mean tail_returns