open Torch
open Types
open Config
open Utils

let eta_n_l config framework xi n l k =
  if n <= 0 || l < 0 || k < 0 then 
    raise (Error (InvalidParameter "n must be positive, l and k must be non-negative"))
  else
    let h_lk = h config (l + k) in
    let sat_factor = saturation_factor framework n in
    config.ca *. sat_factor *. (h_lk ** (1. /. config.r)) *. (float_of_int n ** (-1. /. (2. *. config.theta)))

let refine config framework phi y z xi n l =
  let rec refine_step k x =
    if k > int_of_float (config.theta *. float_of_int l) then x
    else
      let threshold = eta_n_l config framework xi n l k in
      let x_refined = phi y z in
      if Tensor.(to_float0_exn (abs (x_refined - (f xi))) <= f threshold) then
        x_refined
      else
        refine_step (k + 1) x_refined
  in
  refine_step 0 (phi y z)