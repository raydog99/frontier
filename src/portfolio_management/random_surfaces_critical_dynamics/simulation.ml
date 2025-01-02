open Torch
open Types
open Constants
open Utils
open Logging
open Liouville_theory
open Order_parameter
open Analysis

let run_simulation params =
  debug "Starting simulation run";
  try
    let rec simulate area genus order_param step areas genera order_params =
      if step >= params.num_steps then
        (List.rev areas, List.rev genera, List.rev order_params)
      else
        let new_area = evolve_area area genus params.dt in
        let new_genus = evolve_genus genus new_area in
        let new_order_param = evolve_order_parameter order_param new_area params.dt in
        simulate new_area new_genus new_order_param (step + 1)
          (new_area :: areas) (new_genus :: genera) (new_order_param :: order_params)
    in
    let (areas, genera, order_params) = 
      simulate params.initial_area params.initial_genus params.initial_order_parameter 0 [] [] []
    in
    let order_param_tensor = Tensor.of_float1 (Array.of_list order_params) in
    let returns = compute_returns order_param_tensor 1 in
    let hurst_exponent = compute_hurst_exponent returns 2 in
    let multifractal_spectrum = 
      if params.use_multifractal then
        Some (analyze_multifractal_scaling returns 20 100)
      else
        None
    in
    let regime_transitions = analyze_regime_transitions areas genera in
    let autocorrelation = compute_autocorrelation returns params.max_lag in
    let lyapunov_exponent = estimate_lyapunov_exponent order_params in
    let fractal_dimension = compute_fractal_dimension returns params.num_boxes in
    
    info "Simulation run completed successfully";
    {
      areas = areas;
      genera = genera;
      order_parameters = order_params;
      returns = returns;
      hurst_exponent = hurst_exponent;
      multifractal_spectrum = multifractal_spectrum;
      regime_transitions = regime_transitions;
      autocorrelation = autocorrelation;
      lyapunov_exponent = lyapunov_exponent;
      fractal_dimension = fractal_dimension;
      config = params;
    }
  with
  | e ->
      error (Printf.sprintf "Error during simulation: %s" (Printexc.to_string e));
      raise e