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

let create ~eta ~gamma ~sigma_s ~sigma_x ~rho ~lambda ~t ~T ~s ~r_xx ~r_xa ~r_aa ~r_vv ~r_va ~kappa =
  { eta; gamma; sigma_s; sigma_x; rho; lambda; t; T; s; r_xx; r_xa; r_aa; r_vv; r_va; kappa }