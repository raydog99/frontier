type genus = int
type area = float
type background_time = float
type physical_time = float
type conformal_factor = float
type order_parameter = float

type regime = Planar | HigherGenus | Foamy

type cir_params = {
  a: float;
  b: float;
  alpha: float;
}

type multifractal_spectrum = {
  qs: Torch.Tensor.t;
  scaling_exponents: Torch.Tensor.t;
  generalized_hurst_exponents: Torch.Tensor.t;
  alphas: Torch.Tensor.t;
  f_alphas: Torch.Tensor.t;
}

type config = {
  initial_area : float;
  initial_genus : int;
  initial_order_parameter : float;
  num_steps : int;
  dt : float;
  use_multifractal : bool;
  max_lag : int;
  num_boxes : int;
  random_seed : int;
  output_dir : string;
  verbose : bool;
}

type simulation_params = {
  initial_area: float;
  initial_genus: int;
  initial_order_parameter: float;
  num_steps: int;
  dt: float;
  use_multifractal: bool;
  max_lag: int;
  num_boxes: int;
}

type simulation_results = {
  areas: float list;
  genera: int list;
  order_parameters: float list;
  returns: Torch.Tensor.t;
  hurst_exponent: float;
  multifractal_spectrum: multifractal_spectrum option;
  regime_transitions: (regime * float * int) list;
  autocorrelation: Torch.Tensor.t;
  lyapunov_exponent: float;
  fractal_dimension: float;
  config: simulation_params;
}