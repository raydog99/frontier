open Types

val objective_function : Tensor.t -> Tensor.t -> Tensor.t
val differential_evolution : (model_parameters -> float) -> (float * float) array -> int -> int -> model_parameters
val particle_swarm_optimization : (model_parameters -> float) -> (float * float) array -> int -> int -> model_parameters
val calibrate_micro_model : model_parameters -> market_data -> model_parameters
val calibrate_macro_model : model_parameters -> market_data -> model_parameters