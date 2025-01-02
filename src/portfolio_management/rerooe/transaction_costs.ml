type t =
  | Fixed of float
  | Linear of float
  | Nonlinear of (float -> float)

let calculate_cost model volume price =
  match model with
  | Fixed cost -> cost
  | Linear rate -> rate *. abs_float volume *. price
  | Nonlinear f -> f (abs_float volume) *. price

let apply_cost model volume price =
  price -. (calculate_cost model volume price)