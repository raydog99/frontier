open Torch

type date = float
type futures_price = Tensor.t
type index_price = Tensor.t
type volatility = Tensor.t

type model_parameters = {
  kappa: float;
  theta: float;
  chi: float;
  rho: float;
  v0: float;
  a: float;
  beta: float;
  leverage_function: float -> float -> float -> float;
  local_volatility_function: float -> float -> float;
}

type market_data = {
  futures_prices: futures_price array;
  futures_maturities: date array;
  index_price: index_price;
  futures_volatilities: volatility array;
  index_volatility: volatility;
  risk_free_rate: float;
  vanilla_option_prices: Tensor.t;
  vanilla_option_strikes: Tensor.t;
  vanilla_option_maturities: Tensor.t;
}

type option_type = Call | Put

type optimization_result = {
  optimal_params: model_parameters;
  objective_value: float;
}

type futures_contract = {
  maturity: float;
  price: float;
}

type index_calculation_method =
  | StandardRolling
  | EnhancedRolling

type futures_roll_schedule = {
  roll_dates: float array;
  front_contract_weights: float array;
  back_contract_weights: float array;
}

type index_calculation_parameters = {
  roll_schedule: futures_roll_schedule;
  calculation_method: index_calculation_method;
}

type autocallable_contract = {
  observation_dates: float array;
  barriers: float array;
  coupons: float array;
  final_barrier: float;
}

type athena_jet_contract = {
  early_redemption_date: float;
  early_redemption_barrier: float;
  early_redemption_coupon: float;
  maturity: float;
  final_barrier: float;
  participation_rate: float;
}

type path_dependent_option =
  | Autocallable of autocallable_contract
  | AthenaJet of athena_jet_contract
  | DailyBarrierKnockIn of {
      barrier: float;
      maturity: float;
    }

type model_type =
  | Micro
  | Macro

module type MODEL = sig
  val simulate : model_parameters -> market_data -> int -> int -> Tensor.t
  val price_vanilla_option : model_parameters -> market_data -> float -> float -> option_type -> Tensor.t
  val price_path_dependent_option : model_parameters -> market_data -> path_dependent_option -> int -> int -> float
  val compute_delta : model_parameters -> market_data -> path_dependent_option -> float
  val compute_vega : model_parameters -> market_data -> path_dependent_option -> float
end