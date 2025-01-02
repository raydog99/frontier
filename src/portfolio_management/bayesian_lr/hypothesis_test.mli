val test_parameter : 
  Type.posterior_sample list -> int -> float -> float -> Type.hypothesis_test
(** [test_parameter samples idx null_value alpha] performs individual parameter test *)

val test_joint : 
  Type.posterior_sample list -> int list -> float list -> float -> Type.joint_test
(** [test_joint samples indices null_values alpha] performs joint hypothesis test *)