open Types
open Insurance_model
open Sensitivity_analysis

let compute_var
    (model: insurance_model)
    (initial_state: path)
    (t: time)
    (maturity: time)
    (confidence_level: float)
    (num_simulations: int) : float =
  let base_value = (Numerical_methods.solve_pde_adi 
    (thiele_pde_coefficients model 0) 
    (fun _ _ -> 0.0) t maturity initial_state 1000 [|100; 50|]
    [|Dirichlet (fun _ _ -> 0.0); Dirichlet (fun _ _ -> 0.0)|]
    [|Neumann (fun _ _ -> 0.0); Neumann (fun _ _ -> 0.0)|]).value t initial_state in
  
  let losses = Array.init num_simulations (fun _ ->
    let perturbed_model = perturb_model model InterestRate (Random.float 0.02 -. 0.01) in
    let perturbed_model = perturb_model perturbed_model Volatility (Random.float 0.4 -. 0.2) in