
type time = floattype state = int
type path = float array
type non_anticipative_functional = time -> path -> float

type boundary_condition =
  | Dirichlet of (time -> float -> float)
  | Neumann of (time -> float -> float)
  | Mixed of (time -> float -> float * float * float)  (* a * u + b * du/dx = c *)

type markov_process = {
  initial_state: state;
  transition_rates: state -> state -> time -> float;
}

type financial_model = {
  interest_rate: time -> float;
  asset_dynamics: time -> path -> (float array * float array array);  (* drift vector and volatility matrix *)
  maturity: time;
}