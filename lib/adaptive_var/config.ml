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

let create
    ?(checkpoint_interval = None)
    ?(checkpoint_file = None)
    ?(num_threads = 1)
    ?(progress_interval = None)
    ?(max_retries = 3)
    ?(use_adaptive_step_size = false)
    ?(benchmark = false)
    ?(loss_type = Loss_functions.Linear)
    ?(confidence_level = 0.95)
    ?(use_distributed = false)
    ?(master_address = None)
    ?(slave_addresses = [])
    ?(use_adaptive_params = false)
    ?(parallel_chunks = 1)
    ?(random_generator = (module DefaultGenerator : Generator))
    ?(var_method = HistoricalSimulation)
    ?(profiling_enabled = false)
    ?(auto_tune_hyperparameters = false)
    ?(stopping_window_size = 10)
    ?(stopping_tolerance = 1e-6)
    ~alpha ~h0 ~m ~ca ~r ~theta ~gamma ~kappa ~p_star () =
  if alpha <= 0. || alpha >= 1. then raise (Error (InvalidParameter "alpha must be between 0 and 1"));
  if h0 <= 0. then raise (Error (InvalidParameter "h0 must be positive"));
  if m < 2 then raise (Error (InvalidParameter "m must be at least 2"));
  if ca <= 0. then raise (Error (InvalidParameter "ca must be positive"));
  if r <= 1. then raise (Error (InvalidParameter "r must be greater than 1"));
  if theta <= 0. || theta > 1. then raise (Error (InvalidParameter "theta must be between 0 and 1"));
  if gamma <= 0. then raise (Error (InvalidParameter "gamma must be positive"));
  if kappa <= 0. then raise (Error (InvalidParameter "kappa must be positive"));
  if p_star <= 2. then raise (Error (InvalidParameter "p_star must be greater than 2"));
  if num_threads < 1 then raise (Error (InvalidParameter "num_threads must be at least 1"));
  if max_retries < 0 then raise (Error (InvalidParameter "max_retries must be non-negative"));
  if confidence_level <= 0. || confidence_level >= 1. then raise (Error (InvalidParameter "confidence_level must be between 0 and 1"));
  if parallel_chunks < 1 then raise (Error (InvalidParameter "parallel_chunks must be at least 1"));
  if stopping_window_size < 1 then raise (Error (InvalidParameter "stopping_window_size must be at least 1"));
  if stopping_tolerance <= 0. then raise (Error (InvalidParameter "stopping_tolerance must be positive"));
  { alpha; h0; m; ca; r; theta; gamma; kappa; p_star;
    checkpoint_interval; checkpoint_file; num_threads; progress_interval; max_retries;
    use_adaptive_step_size; benchmark; loss_type; confidence_level;
    use_distributed; master_address; slave_addresses; use_adaptive_params;
    parallel_chunks; random_generator; var_method; profiling_enabled;
    auto_tune_hyperparameters; stopping_window_size; stopping_tolerance }

let default = {
  alpha = 0.05;
  h0 = 1.0;
  m = 2;
  ca = 3.0;
  r = 1.5;
  theta = 0.5;
  gamma = 1.0;
  kappa = 0.5;
  p_star = 4.0;
  checkpoint_interval = None;
  checkpoint_file = None;
  num_threads = 1;
  progress_interval = None;
  max_retries = 3;
  use_adaptive_step_size = false;
  benchmark = false;
  loss_type = Loss_functions.Linear;
  confidence_level = 0.95;
  use_distributed = false;
  master_address = None;
  slave_addresses = [];
  use_adaptive_params = false;
  parallel_chunks = 1;
  random_generator = (module DefaultGenerator : Generator);
  var_method = HistoricalSimulation;
  profiling_enabled = false;
  auto_tune_hyperparameters = false;
  stopping_window_size = 10;
  stopping_tolerance = 1e-6;
}