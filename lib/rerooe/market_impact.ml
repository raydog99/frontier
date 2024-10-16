type t =
  | Linear of { permanent: float; temporary: float }
  | SquareRoot of { permanent: float; temporary: float }
  | Exponential of { permanent: float; temporary: float; decay: float }

let calculate_impact model volume price =
  match model with
  | Linear { permanent; temporary } ->
      let perm_impact = permanent *. volume *. price in
      let temp_impact = temporary *. volume *. price in
      (perm_impact, temp_impact)
  | SquareRoot { permanent; temporary } ->
      let perm_impact = permanent *. price *. sqrt (abs_float volume) *. (if volume >= 0. then 1. else -1.) in
      let temp_impact = temporary *. price *. sqrt (abs_float volume) *. (if volume >= 0. then 1. else -1.) in
      (perm_impact, temp_impact)
  | Exponential { permanent; temporary; decay } ->
      let perm_impact = permanent *. price *. (1. -. exp (-. decay *. abs_float volume)) *. (if volume >= 0. then 1. else -1.) in
      let temp_impact = temporary *. price *. (1. -. exp (-. decay *. abs_float volume)) *. (if volume >= 0. then 1. else -1.) in
      (perm_impact, temp_impact)

let apply_impact model volume price =
  let (perm_impact, temp_impact) = calculate_impact model volume price in
  price +. perm_impact +. temp_impact