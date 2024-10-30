open Types

val binomial_log_likelihood : data -> model_params -> float
val poisson_log_likelihood : data -> model_params -> float
val mc_likelihood : data -> model_params -> 
                   int -> float