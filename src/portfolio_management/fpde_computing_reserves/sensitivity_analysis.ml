open Types
open Insurance_model

type parameter = 
  | InterestRate
  | Volatility
  | Participation
  | Mortality

let perturb_model (model: insurance_model) (param: parameter) (perturbation: float) : insurance_model =
  match param with
  | InterestRate ->
      let new_financial_model = {
        model.financial_model with
        interest_rate = (fun t -> model.financial_model.interest_rate t +. perturbation)
      } in
      { model with financial_model = new_financial_model }
  | Volatility ->
      let new_financial_model = {
        model.financial_model with
        asset_dynamics = (fun t omega ->
          let (drift, diffusion) = model.financial_model.asset_dynamics t omega in
          (drift, Array.map (Array.map (fun x -> x *. (1. +. perturbation))) diffusion)
        )
      } in
      { model with financial_model = new_financial_model }
  | Participation ->
      let new_cash_flow = {
        model.cash_flow with
        transition_payments = Array.map (Array.map (fun f t omega -> 
          f t omega *. (1. +. perturbation)
        )) model.cash_flow.transition_payments
      } in
      { model with cash_flow = new_cash_flow }
  | Mortality ->
      let new_markov_process = {
        model.markov_process with
        transition_rates = (fun from_state to_state t ->
          model.markov_process.transition_rates from_state to_state t *. (1. +. perturbation)
        )
      } in
      { model with markov_process = new_markov_process }

let compute_sensitivity 
    (model: insurance_model) 
    (param: parameter) 
    (perturbation: float)
    (initial_state: path)
    (t: time)
    (maturity: time) : float =
  let base_value = (Numerical_methods.solve_pde_adi 
    (thiele_pde_coefficients model 0) 
    (fun _ _ -> 0.0) t maturity initial_state 1000 [|100; 50|]
    [|Dirichlet (fun _ _ -> 0.0); Dirichlet (fun _ _ -> 0.0)|]
    [|Neumann (fun _ _ -> 0.0); Neumann (fun _ _ -> 0.0)|]).value t initial_state in
  
  let perturbed_model = perturb_model model param perturbation in
  let perturbed_value = (Numerical_methods.solve_pde_adi 
    (thiele_pde_coefficients perturbed_model 0) 
    (fun _ _ -> 0.0) t maturity initial_state 1000 [|100; 50|]
    [|Dirichlet (fun _ _ -> 0.0); Dirichlet (fun _ _ -> 0.0)|]
    [|Neumann (fun _ _ -> 0.0); Neumann (fun _ _ -> 0.0)|]).value t initial_state in
  
  (perturbed_value -. base_value) /. (perturbation *. base_value)

let stress_test
    (model: insurance_model)
    (stresses: (parameter * float) list)
    (initial_state: path)
    (t: time)
    (maturity: time) : float =
  let stressed_model = List.fold_left 
    (fun acc_model (param, stress) -> perturb_model acc_model param stress) 
    model stresses in
  
  (Numerical_methods.solve_pde_adi 
    (thiele_pde_coefficients stressed_model 0) 
    (fun _ _ -> 0.0) t maturity initial_state 1000 [|100; 50|]
    [|Dirichlet (fun _ _ -> 0.0); Dirichlet (fun _ _ -> 0.0)|]
    [|Neumann (fun _ _ -> 0.0); Neumann (fun _ _ -> 0.0)|]).value t initial_state