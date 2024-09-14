open Torch
open Types

type diagnostic_result = {
  lb_stat: float;
  lb_p_value: float;
  jb_stat: float;
  jb_p_value: float;
  adf_stat: float;
  adf_p_value: float;
}

val ljung_box_test : Tensor.t -> int -> float * float
val jarque_bera_test : Tensor.t -> float * float
val adf_test : Tensor.t -> float * float
val run_diagnostics : garch_model -> float * float * float * float -> Tensor.t -> diagnostic_result