open Types
open Model
open Optimization
open Logging
open Error_handling
open Performance

let rec payment_dates start_date end_date =
  if start_date >= end_date then []
  else
    let next_date = start_date +. 0.5 in
    if next_date > end_date then [end_date]
    else start_date :: payment_dates next_date end_date

let price_cds params t T recovery_rate order =
  let x0 = params.r0 *. exp (params.alpha1 *. t) in
  let y0 = params.lambda0 *. exp (params.alpha2 *. t) in
  
  let payment_dates = payment_dates t T in
  
  let calculate_leg dates f =
    parallel_map (fun s -> f s) dates
    |> List.fold_left (+.) 0.0
  in
  
  let protection_leg = (1.0 -. recovery_rate) *. 
    calculate_leg payment_dates (fun s -> exp (-.params.alpha2 *. s) *. h params 0.0 x0 y0 s order) in
  
  let premium_leg_1 = calculate_leg payment_dates (fun s -> 
    exp (-.params.alpha2 *. s) *. h params 0.0 x0 y0 s order *. (s -. (max (s -. 0.5) 0.0))) in
  
  let premium_leg_2 = calculate_leg payment_dates (fun ti -> 
    0.5 *. v params 0.0 x0 y0 ti order) in
  
  let premium_leg = premium_leg_1 +. premium_leg_2 in
  
  let spread = protection_leg /. premium_leg in
  debug (Printf.sprintf "CDS spread for T=%.2f: %.6f" T spread);
  spread

let calibrate_ir_model zcb_prices =
  let objective (alpha1, beta1, sigma1) =
    List.fold_left (fun acc (T, price) ->
      let model_price = v0 {alpha1; beta1; sigma1; alpha2=0.; beta2=0.; sigma2=0.; rho=0.; r0=0.; lambda0=0.} 0. 0. 0. T in
      acc +. (model_price -. price) ** 2.
    ) 0. zcb_prices
  in
  
  let initial_simplex = [|
    [|0.1; 0.05; 0.02|];
    [|0.2; 0.05; 0.02|];
    [|0.1; 0.10; 0.02|];
    [|0.1; 0.05; 0.04|]
  |] in
  
  let (result, _) = nelder_mead objective initial_simplex 1e-6 1000 in
  (result.(0), result.(1), result.(2))

let match_zcb_price params x0 T =
  let p_cir = v0 params 0.0 x0 0.0 T in
  let p_approx sigma =
    v0 {params with sigma1 = sigma} 0.0 x0 0.0 T +.
    v1 {params with sigma1 = sigma} 0.0 x0 0.0 T
  in
  
  let objective sigma =
    let diff = p_cir -. p_approx sigma in
    diff *. diff
  in
  
  golden_section_search objective 0. 1. 1e-6

let calibrate_to_market_data params market_data order =
  info "Starting CDS model calibration";
  let (alpha1, beta1, sigma1) = calibrate_ir_model market_data.zcb_prices in
  let x0 = params.r0 *. exp (alpha1 *. 0.0) in
  let sigma_hat = match_zcb_price {params with alpha1; beta1; sigma1} x0 10.0 in
  
  let objective params =
    parallel_map (fun (T, spread) ->
      let model_spread = price_cds params 0.0 T 0.4 order in
      (model_spread -. spread) ** 2.0
    ) market_data.cds_spreads
    |> List.fold_left (+.) 0.0
  in
  
  let initial_simplex = [|
    [|params.alpha2; params.beta2; params.sigma2; params.lambda0; params.rho|];
    [|params.alpha2 *. 1.1; params.beta2; params.sigma2; params.lambda0; params.rho|];
    [|params.alpha2; params.beta2 *. 1.1; params.sigma2; params.lambda0; params.rho|];
    [|params.alpha2; params.beta2; params.sigma2 *. 1.1; params.lambda0; params.rho|];
    [|params.alpha2; params.beta2; params.sigma2; params.lambda0 *. 1.1; params.rho|];
    [|params.alpha2; params.beta2; params.sigma2; params.lambda0; min (params.rho +. 0.1) 1.|];
  |] in
  
  let (result, optimization_time) = time_it (fun () -> nelder_mead objective initial_simplex 1e-6 1000) () in
  let (optimal_params, error) = result in
  
  info (Printf.sprintf "CDS model calibration completed in %.2f seconds" optimization_time);
  { params = {
      params with
      alpha1 = alpha1;
      beta1 = beta1;
      sigma1 = sigma_hat;
      alpha2 = optimal_params.(0);
      beta2 = optimal_params.(1);
      sigma2 = optimal_params.(2);
      lambda0 = optimal_params.(3);
      rho = optimal_params.(4);
    };
    error = error
  }