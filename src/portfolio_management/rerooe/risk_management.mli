type risk_measure =
  | ValueAtRisk of float
  | ConditionalVaR of float
  | ExpectedShortfall of float
  | MaxDrawdown

type risk_limit =
  | AbsoluteLimit of float
  | RelativeLimit of float

type t = {
  measure: risk_measure;
  limit: risk_limit;
}

type advanced_risk_measure =
  | TailValueAtRisk of float
  | ExpectedTail of float
  | ConditionalDrawdown of float

val calculate_risk : Portfolio.t -> risk_measure -> float
val check_risk_limit : Portfolio.t -> t -> bool
val adjust_position : Portfolio.t -> t -> unit
val calculate_advanced_risk : Portfolio.t -> advanced_risk_measure -> float
val adjust_position_advanced : Portfolio.t -> t -> unit