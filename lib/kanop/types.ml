type option_type = Call | Put

type option_style = American | AsianAmerican

type option_params = {
  s0 : float;
  strike : float;
  ttm : float;
  volatility : float;
  risk_free_rate : float;
  dividend_rate : float;
}

type kan_params = {
  input_dim : int;
  hidden_dim : int;
  output_dim : int;
}

exception Invalid_parameter of string