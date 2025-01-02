open Torch
open Types

val log_posterior : data -> model_params -> 
                   prior_params -> Tensor.t
val log_likelihood : data -> model_params -> Tensor.t
val prior : model_params -> prior_params -> Tensor.t