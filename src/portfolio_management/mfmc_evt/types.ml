type distribution_params = 
  | GaussianParams of { mu: float; sigma: float }
  | GumbelParams of { mu: float; beta: float }
  | GEVParams of { location: float; scale: float; shape: float }
  | GPDParams of { scale: float; shape: float }
  | BivariateGaussianParams of { mu1: float; mu2: float; sigma1: float; sigma2: float; rho: float }
  | BivariateGumbelParams of { mu1: float; mu2: float; beta1: float; beta2: float; r: float }

type simulation_params = {
  duration: float;
  time_step: float;
  seed: int;
}

type estimation_method = 
  | Baseline
  | JointMLE
  | MomentMF
  | MarginalMLE

type extreme_value_method =
  | BlockMaxima of int  (* block size *)
  | PeaksOverThreshold of float  (* threshold *)

type mf_error = 
  | InvalidParameters of string
  | FittingError of string
  | SimulationError of string
  | EstimationError of string

exception MFError of mf_error