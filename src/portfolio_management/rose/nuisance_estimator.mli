type nuisance_functions = {
  m_hat: float array;
  f_hat: float array;
  h_hat: float array;
  v_hat: float array array;
}

val estimate_conditional_mean : Types.observation array -> float array
val estimate_nonparametric_component : Types.observation array -> float -> float array
val estimate_all : Types.observation array -> Models.model_type -> nuisance_functions
val check_rates : nuisance_functions -> nuisance_functions -> int -> bool