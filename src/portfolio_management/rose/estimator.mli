val compute_sandwich_variance : Types.observation array -> float array -> float array -> float array -> float
val compute_confidence_interval : float -> float -> float -> float * float
val solve_estimating_equations : Types.observation array -> float array -> 
                               Types.nuisance_estimates array -> float -> int -> float -> float