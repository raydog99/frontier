val create_omega_map : 
  Scm.t -> Scm.t -> (int * int) list -> (Types.intervention -> Types.intervention)
val is_order_preserving : 
  (Types.intervention -> Types.intervention) -> 
  Types.intervention -> Types.intervention -> bool