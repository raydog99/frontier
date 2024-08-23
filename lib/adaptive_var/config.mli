open Types
open Random_generator
open Var_estimation

type t = {
  alpha : float;
  h0 : float;
  m : int;
  ca : float;
  r : float;
  theta : float;
  gamma : float;
  kappa : float;
  p_star : float;
  checkpoint_interval : int option;
  checkpoint_file : string option;
  num_threads : int;
  progress_interval : int option;
  max_retries : int;
  use_adaptive_step_size : bool;
  benchmark : bool;
  loss_type : Loss_functions.loss_type;
  confidence_level : float;
  use_distributed : bool;
  master_address : string option;
  slave_addresses : string list;
  use_adaptive_params : bool;
  parallel_chunks : int;
  random_generator : (module Generator);
  var_method : method_t;
  profiling_enabled : bool;
  auto_tune_hyperparameters : bool;
  stopping_window_size : int;
  stopping_tolerance : float;
}

val create :
  ?checkpoint_interval:int option ->
  ?checkpoint_file:string option ->
  ?num_threads:int ->
  ?progress_interval:int option ->
  ?max_retries:int ->
  ?use_adaptive_step_size:bool ->
  ?benchmark:bool ->
  ?loss_type:Loss_functions.loss_type ->
  ?confidence_level:float ->
  ?use_distributed:bool ->
  ?master_address:string option ->
  ?slave_addresses:string list ->
  ?use_adaptive_params:bool ->
  ?parallel_chunks:int ->
  ?random_generator:(module Generator) ->
  ?var_method:method_t ->
  ?profiling_enabled:bool ->
  ?auto_tune_hyperparameters:bool ->
  ?stopping_window_size:int ->
  ?stopping_tolerance:float ->
  alpha:float ->
  h0:float ->
  m:int ->
  ca:float ->
  r:float ->
  theta:float ->
  gamma:float ->
  kappa:float ->
  p_star:float ->
  unit ->
  t

val default : t