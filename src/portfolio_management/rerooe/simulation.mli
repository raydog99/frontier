val run_single_path : Simulation_config.t -> (float * float * float) list
val run_multiple_paths : Simulation_config.t -> (float * float * float) list list
val calculate_statistics : (float * float * float) list list -> (float * float * float) list * (float * float * float) list