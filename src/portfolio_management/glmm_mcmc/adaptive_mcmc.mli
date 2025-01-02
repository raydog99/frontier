open Torch
open Types

  type adaptive_state = {
    mcmc_state: mcmc_state;
    covariance: Tensor.t;
    mean: Tensor.t;
    samples: model_params list;
    adaptation_count: int;
  }

  val update_moments : adaptive_state -> adaptive_state

  val adapt_step : data -> adaptive_state -> adaptive_state

  val run : data -> 
           adaptive_state -> 
           int ->  (* n_adapt *)
           int ->  (* n_samples *)
           model_params list