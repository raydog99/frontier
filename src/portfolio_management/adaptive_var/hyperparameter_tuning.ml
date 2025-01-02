open Torch
open Types
open Config
open Admlsa

type hyperparameters = {
  theta: float;
  r: float;
  ca: float;
  m: int;
}

let objective_function config framework phi eps l params =
  let updated_config = { config with
    theta = params.theta;
    r = params.r;
    ca = params.ca;
    m = params.m;
  } in
  Lwt_main.run (admlsa updated_config framework phi eps l)

let bayesian_optimization config framework phi eps l =
  let bounds = [
    ("theta", (0.1, 1.0));
    ("r", (1.1, 2.0));
    ("ca", (1.0, 5.0));
    ("m", (2, 10));
  ] in
  let initial_points = 10 in
  let max_iterations = 50 in
  
  let module BO = Owl_opt.Bayesian_optimization.Make (Owl_opt.Inputs.Float) in
  let optimizer = BO.init ~bounds ~initial_points in
  
  let rec optimize iter optimizer best_params best_value =
    if iter >= max_iterations then (best_params, best_value)
    else
      let params, optimizer = BO.suggest optimizer in
      let value = objective_function config framework phi eps l {
        theta = params.(0);
        r = params.(1);
        ca = params.(2);
        m = int_of_float params.(3);
      } in
      let optimizer = BO.update optimizer params value in
      if value < best_value then
        optimize (iter + 1) optimizer { theta = params.(0); r = params.(1); ca = params.(2); m = int_of_float params.(3) } value
      else
        optimize (iter + 1) optimizer best_params best_value
  in
  
  let initial_params = {
    theta = config.theta;
    r = config.r;
    ca = config.ca;
    m = config.m;
  } in
  let initial_value = objective_function config framework phi eps l initial_params in
  optimize 0 optimizer initial_params initial_value