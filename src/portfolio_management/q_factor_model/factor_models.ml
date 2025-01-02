open Torch

type factor = {
  name: string;
  data: Tensor.t;
}

let create_q_factor_model market_data size_data investment_data roe_data =
  FactorModel.create "Q-factor" [
    Factor.create "Market" market_data;
    Factor.create "Size" size_data;
    Factor.create "Investment" investment_data;
    Factor.create "ROE" roe_data;
  ]

let create_q5_model market_data size_data investment_data roe_data expected_growth_data =
  FactorModel.create "Q5" [
    Factor.create "Market" market_data;
    Factor.create "Size" size_data;
    Factor.create "Investment" investment_data;
    Factor.create "ROE" roe_data;
    Factor.create "Expected Growth" expected_growth_data;
  ]

let create_fama_french_5_factor_model market_data smb_data hml_data rmw_data cma_data =
  FactorModel.create "Fama-French 5-factor" [
    Factor.create "Market" market_data;
    Factor.create "SMB" smb_data;
    Factor.create "HML" hml_data;
    Factor.create "RMW" rmw_data;
    Factor.create "CMA" cma_data;
  ]