type t = { risk_aversion: float }

let create risk_aversion = { risk_aversion }

let calculate t wealth =
wealth -. 0.5 *. t.risk_aversion *. wealth *. wealth

let risk_aversion t = t.risk_aversion