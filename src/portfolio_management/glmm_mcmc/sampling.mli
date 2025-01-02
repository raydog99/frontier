module Rejection : sig
  val adaptive_rejection_sample : (float -> float) -> float
  val slice_sample : (float -> float) -> float -> float -> float
end

module Polya_Gamma : sig
  type pg_state = {
    h: float;
    z: float;
  }
  
  val sample : pg_state -> float
  val sample_truncated : pg_state -> int -> float
  val approximate_sample : pg_state -> float
end