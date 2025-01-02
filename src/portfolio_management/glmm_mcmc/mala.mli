open Types

val step : data -> mcmc_state -> float -> mcmc_state
val preconditioned_step : data -> mcmc_state -> 
                         Tensor.t -> float -> mcmc_state
val manifold_step : data -> mcmc_state -> 
                   float -> mcmc_state