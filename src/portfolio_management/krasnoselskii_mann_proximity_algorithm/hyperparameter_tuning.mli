type hyperparameters = {
  alpha: float;
  max_iterations: int;
  tolerance: float;
  rel_tolerance: float;
  stagnation_window: int;
  adaptive_step: bool;
}

val random_search : int -> (hyperparameters -> float) -> hyperparameters -> (hyperparameters * float)
val grid_search : (hyperparameters -> float) -> hyperparameters -> (hyperparameters * float)