open Torch
open Types
open Constants
open Utils
open Logging

let evolve_order_parameter current_value area dt =
  let drift = 0.0 in
  let volatility = sqrt area in
  geometric_brownian_motion current_value drift volatility dt

let compute_returns order_parameter_series time_horizon =
  let n = Tensor.shape order_parameter_series |> List.hd in
  let returns = Tensor.sub (Tensor.narrow order_parameter_series 0 time_horizon (n - time_horizon))
                           (Tensor.narrow order_parameter_series 0 0 (n - time_horizon)) in
  returns

let compute_moment returns n =
  let powered_returns = Tensor.pow returns (float_of_int n) in
  Tensor.mean powered_returns

let compute_hurst_exponent returns n =
  let log_time_horizon = Tensor.log (Tensor.arange 1 (Tensor.shape returns |> List.hd |> float_of_int) 1) in
  let log_moment = Tensor.log (compute_moment returns n) in
  let slope, _ = linear_regression log_time_horizon log_moment in
  slope /. float_of_int n

let equilibrium_return_distribution returns =
  let std_dev = Tensor.std returns in
  let degrees_of_freedom = 4.0 +. 2.0 /. float_of_int minimal_model_m in
  StudentsT (std_dev, degrees_of_freedom)

let generalized_hyperbolic_distribution returns genus =
  if genus = 0 then
    equilibrium_return_distribution returns
  else
    let mean = Tensor.mean returns in
    let variance = Tensor.var returns in
    VarianceGamma (mean, variance, float_of_int genus)

let analyze_multifractal_scaling returns max_q max_dt =
  let compute_structure_function returns q dt =
    let abs_returns = Tensor.abs returns in
    let powered_returns = Tensor.pow abs_returns (Tensor.of_float q) in
    Tensor.mean powered_returns
  in

  let estimate_scaling_exponent returns q max_dt =
    let dts = Tensor.arange 1 (float_of_int max_dt) 1.0 in
    let structure_functions = Tensor.stack (List.init max_dt (fun dt -> 
      compute_structure_function returns q (float_of_int (dt + 1)))) in
    let log_dts = Tensor.log dts in
    let log_structure_functions = Tensor.log structure_functions in
    let slope, _ = linear_regression log_dts log_structure_functions in
    slope
  in

  let qs = Tensor.arange (-. (float_of_int max_q)) (float_of_int max_q) 0.5 in
  let scaling_exponents = Tensor.stack (List.init (Tensor.shape qs |> List.hd) (fun i ->
    estimate_scaling_exponent returns (Tensor.get qs [i]) max_dt)) in
  
  let generalized_hurst_exponents = Tensor.div scaling_exponents qs in
  
  let alphas = Tensor.sub (Tensor.neg (Tensor.mul qs (Tensor.grad generalized_hurst_exponents))) generalized_hurst_exponents in
  let f_alphas = Tensor.add (Tensor.mul qs alphas) (Tensor.mul qs generalized_hurst_exponents) in

  (qs, scaling_exponents, generalized_hurst_exponents, alphas, f_alphas)