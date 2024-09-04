open Torch
open Types
open Dns_fr_model

type shock = {
  magnitude: float;
  duration: int;
  affected_maturities: int list;
}

val apply_shock : Tensor.t -> shock -> Tensor.t

val stress_test_dns_fr : dns_fr_model -> state_vector -> Tensor.t -> Tensor.t -> shock -> int -> observation_vector list * observation_vector list

val calculate_stress_impact : observation_vector list -> observation_vector list -> Tensor.t list

val var_stress_test : dns_fr_model -> state_vector -> Tensor.t -> Tensor.t -> shock -> int -> float -> int -> Tensor.t

val expected_shortfall_stress_test : dns_fr_model -> state_vector -> Tensor.t -> Tensor.t -> shock -> int -> float -> int -> Tensor.t