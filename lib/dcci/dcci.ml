open Types
open Torch
open Lwt

val create_model : model_type -> (module MODEL)
val calibrate_model : model_type -> model_parameters -> market_data -> model_parameters Lwt.t
val simulate : (module MODEL) -> model_parameters -> market_data -> int -> int -> Tensor.t Lwt.t
val price_vanilla_option : (module MODEL) -> model_parameters -> market_data -> float -> float -> option_type -> (Tensor.t * Tensor.t) Lwt.t
val price_path_dependent_option : (module MODEL) -> model_parameters -> market_data -> path_dependent_option -> int -> int -> float Lwt.t
val compare_models : model_parameters -> model_parameters -> market_data -> path_dependent_option -> (float * float * float) Lwt.t
val analyze_model_differences : model_parameters -> model_parameters -> market_data -> path_dependent_option list -> (float * float * float) Lwt.t
val run_analysis : market_data -> path_dependent_option list -> float list -> float list -> ((float * float * float * (float * float * float * float)) * (float * float * float * float) list list) Lwt.t
val generate_report : ((float * float * float * (float * float * float * float)) * (float * float * float * float) list list) -> string -> unit