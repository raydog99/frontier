type t =
  | TWAP
  | VWAP of { volume_curve: float -> float }
  | Optimal

let twap model params =
  let x = Model.get_x model in
  let remaining_time = params.Params.T -. params.Params.t in
  if remaining_time > 0. then
    -. x /. remaining_time
  else
    0.

let vwap volume_curve model params =
  let x = Model.get_x model in
  let t = params.Params.t in
  let T = params.Params.T in
  let remaining_volume = volume_curve T -. volume_curve t in
  if remaining_volume > 0. then
    -. x *. (volume_curve (t +. 0.01) -. volume_curve t) /. (0.01 *. remaining_volume)
  else
    0.

let optimal model params vf =
  let x = Model.get_x model in
  let h2 = Value_function.get_h2 vf in
  let h1 = Value_function.get_h1 vf in
  -. (h2 *. x +. h1) /. (2. *. params.Params.eta)

let execute strategy model params vf dt model_type =
  match strategy with
  | TWAP -> twap model params
  | VWAP { volume_curve } -> vwap volume_curve model params
  | Optimal -> optimal model params vf