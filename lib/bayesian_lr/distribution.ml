let normal_cdf x mu sigma =
  0.5 *. (1. +. erf ((x -. mu) /. (sigma *. sqrt 2.)))

let normal_ppf p mu sigma =
  mu +. sigma *. sqrt 2. *. erfi (2. *. p -. 1.)

let chi_square_cdf x df =
  let gamma_p = function
    | a, x when x < 0. -> 0.
    | a, x when x = 0. -> 0.
    | a, x ->
      let rec sum_term n acc last_term =
        let term = last_term *. x /. (a +. float_of_int n) in
        if abs_float term < 1e-10 *. acc then acc
        else sum_term (n + 1) (acc +. term) term
      in
      let first_term = 1. /. exp (log_gamma a) in
      first_term *. exp (a *. log x -. x) *. sum_term 0 1. 1.
  in
  gamma_p (df /. 2., x /. 2.)