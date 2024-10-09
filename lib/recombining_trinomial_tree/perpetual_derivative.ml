type t = {
  tree: Tree.t;
  gamma: float;
}

let create tree =
  let gamma = -2. *. tree.r /. (tree.sigma ** 2.) in
  { tree; gamma }

let price derivative s =
  s ** derivative.gamma

let drift derivative =
  let tree = derivative.tree in
  (1. +. derivative.gamma) *. tree.r -. derivative.gamma *. tree.sigma ** 2. /. 2.

let volatility derivative =
  -. derivative.gamma *. derivative.tree.sigma

let delta derivative s =
  derivative.gamma *. s ** (derivative.gamma -. 1.)

let gamma derivative s =
  derivative.gamma *. (derivative.gamma -. 1.) *. s ** (derivative.gamma -. 2.)

let theta _derivative _s =
  0.

let vega derivative s =
  let tree = derivative.tree in
  let dgamma_dsigma = 4. *. tree.r /. (tree.sigma ** 3.) in
  -. s ** derivative.gamma *. log s *. dgamma_dsigma

let rho derivative s =
  let tree = derivative.tree in
  let dgamma_dr = -2. /. (tree.sigma ** 2.) in
  s ** derivative.gamma *. log s *. dgamma_dr