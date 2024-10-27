type model_type =
  | PartiallyLinear
  | GeneralizedPartiallyLinear of (float -> float)

let compute_influence model_type obs nuisance theta =
  match model_type with
  | PartiallyLinear ->
      let x_centered = obs.x -. nuisance.m_hat in
      let resid = obs.y -. (theta *. obs.x) -. nuisance.f_hat in
      (x_centered *. resid, x_centered)

  | GeneralizedPartiallyLinear g ->
      let x_centered = obs.x -. nuisance.m_hat in
      let mu = g(theta *. obs.x +. nuisance.f_hat) in
      let g_derivative = 1.0 in 
      let resid = obs.y -. mu in
      (x_centered *. resid *. g_derivative, x_centered *. g_derivative)