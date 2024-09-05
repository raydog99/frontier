open Types
open Functional_ito

type pde_coefficients = {
  drift: non_anticipative_functional;
  diffusion: non_anticipative_functional;
  rate: non_anticipative_functional;
  source: non_anticipative_functional;
}

let create_pde_coefficients
    (drift: non_anticipative_functional)
    (diffusion: non_anticipative_functional)
    (rate: non_anticipative_functional)
    (source: non_anticipative_functional) : pde_coefficients =
  { drift; diffusion; rate; source }

let path_dependent_pde_operator
    (coeffs: pde_coefficients)
    (f: Functional_ito.functional)
    (t: time)
    (omega: path) : float =
  let df_dt = Functional_ito.horizontal_derivative f t omega in
  let df_dx = Functional_ito.vertical_derivative f t omega in
  let d2f_dx2 = Functional_ito.second_vertical_derivative f t omega in
  
  df_dt +. 
  coeffs.drift t omega *. df_dx +.
  0.5 *. (coeffs.diffusion t omega ** 2.0) *. d2f_dx2 -.
  coeffs.rate t omega *. f.value t omega +.
  coeffs.source t omega