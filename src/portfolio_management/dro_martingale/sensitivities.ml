open Torch
open Utils

let g_derivative mu x1 x2 =
  let eps = 1e-6 in
  let g_plus = Payoffs.forward_start_call (Tensor.(x1 + float eps)) x2 in
  let g_minus = Payoffs.forward_start_call (Tensor.(x1 - float eps)) x2 in
  Tensor.((g_plus - g_minus) / (float (2. *. eps)))

let distributionally_robust_g mu r constraints =
  let open Tensor in
  let n = 1000 in
  let samples = Tensor.randn [n; 2] in
  let x1 = Tensor.(mu + (r / sqrt (float 2.)) * get samples [Some 0; Some 0]) in
  let x2 = Tensor.(mu + (r / sqrt (float 2.)) * get samples [Some 0; Some 1]) in
  let x1, x2 = 
    match constraints with
    | `Martingale -> Martingale.project x1 x2
    | `Marginal mu1 -> Marginal.project mu1 x1 x2
    | `Both (mu1, _) -> 
        let x1, x2 = Marginal.project mu1 x1 x2 in
        Martingale.project x1 x2
  in
  let g_values = Payoffs.forward_start_call x1 x2 in
  Tensor.(max g_values)

let sensitivity mu constraints =
  let r = Tensor.float 1e-3 in
  let g_r = distributionally_robust_g mu r constraints in
  let g_0 = Payoffs.forward_start_call (Tensor.select mu 0) (Tensor.select mu 1) in
  Tensor.((g_r - g_0) / r)

let adapted_sensitivity mu constraints =
  let r = Tensor.float 1e-3 in
  let n = 1000 in
  let samples = Tensor.randn [n; 2] in
  let x1 = Tensor.(select mu 0 + (r / sqrt (float 2.)) * get samples [Some 0; Some 0]) in
  let x2 = Tensor.(select mu 1 + (r / sqrt (float 2.)) * get samples [Some 0; Some 1]) in
  let x1, x2 = 
    match constraints with
    | `Martingale -> Martingale.project x1 x2
    | `Marginal mu1 -> Marginal.project mu1 x1 x2
    | `Both (mu1, _) -> 
        let x1, x2 = Marginal.project mu1 x1 x2 in
        Martingale.project x1 x2
  in
  let g_values = Payoffs.forward_start_call x1 x2 in
  let g_r = Tensor.(max g_values) in
  let g_0 = Payoffs.forward_start_call (Tensor.select mu 0) (Tensor.select mu 1) in
  Tensor.((g_r - g_0) / r)

let forward_start_sensitivity mu =
  let dx1 = g_derivative mu (Tensor.select mu 0) (Tensor.select mu 1) in
  let dx2 = Tensor.(float 1. - dx1) in
  Tensor.stack [dx1; dx2] ~dim:0

let forward_start_adapted_sensitivity mu =
  let dx1 = g_derivative mu (Tensor.select mu 0) (Tensor.select mu 1) in
  let dx2 = Tensor.(float 1. - dx1) in
  let e_dx1 = Tensor.(mean dx1) in
  Tensor.stack [e_dx1; dx2] ~dim:0

let martingale_sensitivity_formula mu =
  let x1 = Tensor.select mu 0 in
  let x2 = Tensor.select mu 1 in
  let dx1_g = g_derivative mu x1 x2 in
  let e_dx1_g = Tensor.(mean dx1_g) in
  let var_dx1_g = Tensor.(mean (dx1_g - e_dx1_g) ** float 2.) in
  Tensor.(sqrt (float 0.5 * e_dx1_g ** float 2. + var_dx1_g))

let adapted_martingale_sensitivity_formula mu =
  let x1 = Tensor.select mu 0 in
  let x2 = Tensor.select mu 1 in
  let dx1_g = g_derivative mu x1 x2 in
  let e_dx1_g = Tensor.(mean dx1_g) in
  let var_dx2_g = Tensor.(mean (float 1. - dx1_g - (float 1. - e_dx1_g)) ** float 2.) in
  Tensor.(sqrt (float 0.5 * e_dx1_g ** float 2. + var_dx2_g))