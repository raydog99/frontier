open Types
open Config

type t = framework

let epsilon = function
  | Lipschitz_gradient -> fun h -> h ** (default.p_star /. (2. *. (1. +. default.p_star)))
  | Holder_density -> fun h -> sqrt h *. sqrt (abs_float (log h))
  | Subexponential_deviation -> fun h -> sqrt h

let optimal_theta = function
  | Lipschitz_gradient -> (default.p_star /. 2. -. 1.) /. (default.p_star /. 2. +. 1.)
  | Holder_density | Subexponential_deviation -> 1.

let optimal_complexity = function
  | Lipschitz_gradient -> fun eps -> eps ** (-. (2. -. 2. /. default.p_star))
  | Holder_density -> fun eps -> eps ** (-2.) *. (abs_float (log eps)) ** (7. /. 2.)
  | Subexponential_deviation -> fun eps -> eps ** (-2.) *. (abs_float (log eps)) ** (5. /. 2.)