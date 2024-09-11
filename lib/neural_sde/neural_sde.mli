open Types

type t

val create :
    hidden_dim:int ->
    rho:float ->
    r:float ->
    s0:float ->
    prior_std:float ->
    t

val simulate :
    t ->
    num_paths:int ->
    num_steps:int ->
    dt:float ->
    measure:measure ->
    Tensor.t * Tensor.t

val price_option_with_control_variate :
    t ->
    option_data:option_data ->
    num_paths:int ->
    float

val loss :
    t ->
    option_data:option_data list ->
    num_paths:int ->
    float

val time_series_log_likelihood :
    t ->
    time_series:float list ->
    dt:float ->
    float

val bayesian_calibrate :
    t ->
    option_data:option_data list ->
    time_series:float list ->
    num_epochs:int ->
    step_size:float ->
    sigma:float ->
    num_paths:int ->
    t