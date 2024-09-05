open Types

type cash_flow = {
  sojourn_payments: non_anticipative_functional array;
  transition_payments: (state -> state -> non_anticipative_functional) array array;
}

type insurance_model = {
  markov_process: markov_process;
  cash_flow: cash_flow;
  financial_model: financial_model;
}

let create_insurance_model 
    (markov_process : markov_process)
    (cash_flow : cash_flow)
    (financial_model : financial_model) : insurance_model =
  { markov_process; cash_flow; financial_model }

let thiele_pde_coefficients
    (model : insurance_model)
    (state : state) : Path_dependent_pde.pde_coefficients =
  let drift t omega = fst (model.financial_model.asset_dynamics t omega) in
  let diffusion t omega = snd (model.financial_model.asset_dynamics t omega) in
  let rate t = model.financial_model.interest_rate t in
  let source t omega =
    let cf = model.cash_flow in
    let mp = model.markov_process in
    
    (* Sojourn payments *)
    let sojourn_term = cf.sojourn_payments.(state) t omega in
    
    (* Transition payments and reserves *)
    let transition_term = 
      Array.fold_left (fun sum j ->
        if j <> state then
          let mu_ij = mp.transition_rates state j t in
          let h_ij = cf.transition_payments.(state).(j) t omega in
          sum +. mu_ij *. (h_ij +. (Numerical_methods.solve_pde_adi 
            (thiele_pde_coefficients { model with markov_process = { model.markov_process with initial_state = j } } j)
            (fun _ _ -> 0.0) t model.financial_model.maturity omega 1000 [|100; 50|]
            [|Dirichlet (fun _ _ -> 0.0); Dirichlet (fun _ _ -> 0.0)|]
            [|Neumann (fun _ _ -> 0.0); Neumann (fun _ _ -> 0.0)|]).value t omega)
        else
          sum
      ) 0.0 (Array.init (Array.length cf.sojourn_payments) (fun i -> i))
    in
    
    sojourn_term +. transition_term
  in
  Path_dependent_pde.create_pde_coefficients drift diffusion rate source