let normal_cdf x =
  let a1 = 0.254829592 in
  let a2 = -0.284496736 in
  let a3 = 1.421413741 in
  let a4 = -1.453152027 in
  let a5 = 1.061405429 in
  let p = 0.3275911 in
  let sign = if x < 0. then -1. else 1. in
  let x = abs_float x /. sqrt 2. in
  let t = 1. /. (1. +. p *. x) in
  let y = 1. -. (((((a5 *. t +. a4) *. t) +. a3) *. t +. a2) *. t +. a1) *. t *. exp (-. x *. x) in
  0.5 *. (1. +. sign *. y)

let d1 s x t r sigma =
  (log (s /. x) +. (r +. 0.5 *. sigma *. sigma) *. t) /. (sigma *. sqrt t)

let d2 s x t r sigma =
  d1 s x t r sigma -. sigma *. sqrt t

let call_price s x t r sigma =
  let d1 = d1 s x t r sigma in
  let d2 = d2 s x t r sigma in
  s *. normal_cdf d1 -. x *. exp (-. r *. t) *. normal_cdf d2

let put_price s x t r sigma =
  let d1 = d1 s x t r sigma in
  let d2 = d2 s x t r sigma in
  x *. exp (-. r *. t) *. normal_cdf (-. d2) -. s *. normal_cdf (-. d1)

let implied_volatility market_price s x t r is_call =
  let rec newton_raphson sigma max_iter tol =
    if max_iter = 0 then sigma
    else
      let price = if is_call then call_price s x t r sigma else put_price s x t r sigma in
      let vega = s *. sqrt t *. exp (-. 0.5 *. (d1 s x t r sigma) ** 2.) /. sqrt (2. *. Float.pi) in
      let diff = price -. market_price in
      if abs_float diff < tol then sigma
      else
        let new_sigma = sigma -. diff /. vega in
        newton_raphson new_sigma (max_iter - 1) tol
  in
  newton_raphson 0.5 100 1e-6

let delta s x t r sigma is_call =
  let d1 = d1 s x t r sigma in
  if is_call then normal_cdf d1 else normal_cdf d1 -. 1.

let gamma s x t r sigma =
  let d1 = d1 s x t r sigma in
  exp (-. 0.5 *. d1 *. d1) /. (s *. sigma *. sqrt t *. sqrt (2. *. Float.pi))

let theta s x t r sigma is_call =
  let d1 = d1 s x t r sigma in
  let d2 = d2 s x t r sigma in
  let term1 = -. (s *. normal_cdf d1 *. sigma) /. (2. *. sqrt t) in
  let term2 = r *. x *. exp (-. r *. t) *. normal_cdf d2 in
  let term3 = r *. s *. normal_cdf d1 in
  if is_call then term1 -. term2 +. term3 else term1 +. term2 -. term3

let vega s x t r sigma =
  let d1 = d1 s x t r sigma in
  s *. sqrt t *. exp (-. 0.5 *. d1 *. d1) /. sqrt (2. *. Float.pi)