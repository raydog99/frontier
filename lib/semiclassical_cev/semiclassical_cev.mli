type parameters = {
  sigma: float;    (* Volatility *)
  alpha: float;    (* Elasticity parameter *)
  mu: float;       (* Drift *)
  r: float;        (* Risk-free rate *)
}

type classical_path = {
  x: float -> float;  (* Position function *)
  p: float -> float;  (* Momentum function *)
}

type feller_state = {
  x: float;
  mu_x: float;
  sigma_x: float;
  a: float;
  b: float;
  jacobian: float;
}

val normal_cdf : float -> float
val erf : float -> float
val integrate_adaptive : (float -> float) -> float -> float -> float -> float
val call_price : parameters -> float -> float -> float -> float
val call_delta : parameters -> float -> float -> float -> float
val call_gamma : parameters -> float -> float -> float -> float
val to_feller : parameters -> float -> feller_state
val from_feller : parameters -> float -> float
val verify_transform : parameters -> float -> bool
val solve_ode : (float -> float -> float) -> float -> float -> float -> float -> float
val integrate : (float -> float) -> float -> float -> int -> float
val find_classical_path : parameters -> float -> float -> float -> classical_path
val compute_action : parameters -> float -> float -> float -> float
val kernel : parameters -> float -> float -> float -> float
val european_call : parameters -> float -> float -> float -> float
val validate_parameters : parameters -> (parameters, string) result