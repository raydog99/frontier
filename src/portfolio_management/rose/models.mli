type model_type =
  | PartiallyLinear
  | GeneralizedPartiallyLinear of (float -> float)

val compute_influence : model_type -> Types.observation -> Types.nuisance_estimates -> float -> float * float