open Performance_evaluation

val train_and_evaluate : 
  Sac_agent.t -> 
  Rl_environment.t -> 
  Benchmark_models.t list -> 
  int -> 
  int -> 
  float list * float list

val evaluate_agent : 
  Sac_agent.t -> 
  Rl_environment.t -> 
  performance

val evaluate_benchmark : 
  Benchmark_models.t -> 
  Rl_environment.t -> 
  performance

val compare_performance : 
  performance -> 
  performance -> 
  unit

val run_experiment : 
  string -> 
  int -> 
  int -> 
  float list * float list