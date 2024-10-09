type t =
  | Geometric_Brownian_Motion of float * float
  | Ornstein_Uhlenbeck of float * float * float
  | Heston of float * float * float * float * float
  | Merton_Jump_Diffusion of float * float * float * float

val simulate : t -> float -> int -> float -> float array