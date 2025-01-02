type t = {
  mutable theta: float;
  mutable r: float;
  mutable ca: float;
  mutable m: int;
}

let create ~theta ~r ~ca ~m = { theta; r; ca; m }

let update t ~performance ~convergence_rate ~error ~level =
  t.theta <- t.theta *. (1. +. 0.1 *. (performance -. 0.5));
  t.r <- t.r *. (1. +. 0.05 *. (convergence_rate -. 1.));
  t.ca <- t.ca *. (1. +. 0.1 *. (error -. 0.1));
  t.m <- max 2 (t.m + (if performance > 0.8 then 1 else if performance < 0.2 then -1 else 0))

let get_params t = (t.theta, t.r, t.ca, t.m)