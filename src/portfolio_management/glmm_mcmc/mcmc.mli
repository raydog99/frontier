open Types

val monte_carlo_em : data -> model_params -> 
                    int -> float -> model_params

val mc_maximum_likelihood : data -> model_params -> 
                          model_params -> int -> float
                          
module Adaptive : sig
  val step : data -> mcmc_state -> 
            model_params list -> mcmc_state
end