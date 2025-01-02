type t = {
  eta: float;
  gamma: float;
  sigma_s: float;
  sigma_x: float;
  rho: float;
  lambda: float;
  t: float;
  T: float;
  s: float;
  r_xx: float;
  r_xa: float;
  r_aa: float;
  r_vv: float;
  r_va: float;
  kappa: float;
}

val create :
  eta:float ->
  gamma:float ->
  sigma_s:float ->
  sigma_x:float ->
  rho:float ->
  lambda:float ->
  t:float ->
  T:float ->
  s:float ->
  r_xx:float ->
  r_xa:float ->
  r_aa:float ->
  r_vv:float ->
  r_va:float ->
  kappa:float ->
  t