open Torch
open Utils
open Models
open Sensitivities
open Optimal_stopping
open Optimization

let analyze_model model_type sigma t =
  let s0 = Tensor.float 1. in
  let mu = match model_type with
    | `BlackScholes -> black_scholes s0 sigma t
    | `Bachelier -> bachelier s0 sigma t
  in
  let sensitivity = sensitivity mu `Martingale in
  let adapted_sensitivity = adapted_sensitivity mu `Martingale in
  let forward_start_sensitivity = forward_start_sensitivity mu in
  let forward_start_adapted_sensitivity = forward_start_adapted_sensitivity mu in
  let martingale_sensitivity = martingale_sensitivity_formula mu in
  let adapted_martingale_sensitivity = adapted_martingale_sensitivity_formula mu in
  {
    model = model_type;
    sigma = Tensor.to_float0_exn sigma;
    t = Tensor.to_float0_exn t;
    sensitivity;
    adapted_sensitivity;
    forward_start_sensitivity;
    forward_start_adapted_sensitivity;
    martingale_sensitivity;
    adapted_martingale_sensitivity;
  }

let compare_models sigma_range t_range =
  let models = [`BlackScholes; `Bachelier] in
  List.map (fun model ->
    List.map (fun sigma ->
      List.map (fun t ->
        analyze_model model (Tensor.float sigma) (Tensor.float t)
      ) t_range
    ) sigma_range
  ) models

let optimal_hedging_strategy payoff mu constraints =
  let hedge = optimal_hedge payoff mu constraints in
  let semi_static_hedge = optimal_semi_static_hedge payoff mu constraints in
  {
    dynamic_hedge = hedge;
    semi_static_hedge;
  }

let analyze_optimal_stopping l1 l2 mu constraints =
  let v = value_function l1 l2 in
  let stop_time = optimal_stopping_time l1 l2 (Tensor.select mu 0) (Tensor.select mu 1) in
  let sensitivity = sensitivity mu l1 l2 constraints in
  let adapted_sensitivity = adapted_sensitivity mu l1 l2 constraints in
  let formula_sensitivity = optimal_stopping_sensitivity_formula mu l1 l2 in
  {
    value = v (Tensor.select mu 0) (Tensor.select mu 1);
    stopping_time = stop_time;
    sensitivity;
    adapted_sensitivity;
    formula_sensitivity;
  }

let run_error_analysis model_type sigma t num_trials =
  error_analysis model_type sigma t num_trials

let print_analysis result =
  Printf.printf "Model: %s\n" (match result.model with `BlackScholes -> "Black-Scholes" | `Bachelier -> "Bachelier");
  Printf.printf "Sigma: %f, T: %f\n" result.sigma result.t;
  Printf.printf "Sensitivity: "; print_tensor result.sensitivity;
  Printf.printf "Adapted Sensitivity: "; print_tensor result.adapted_sensitivity;
  Printf.printf "Forward Start Sensitivity: "; print_tensor result.forward_start_sensitivity;
  Printf.printf "Forward Start Adapted Sensitivity: "; print_tensor result.forward_start_adapted_sensitivity;
  Printf.printf "Martingale Sensitivity Formula: "; print_tensor result.martingale_sensitivity;
  Printf.printf "Adapted Martingale Sensitivity Formula: "; print_tensor result.adapted_martingale_sensitivity;
  Printf.printf "\n"