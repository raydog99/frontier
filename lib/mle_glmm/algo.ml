open Types
open Torch
open MatrixOps

type state = {
  beta: Tensor.t;          (* Fixed effects parameters *)
  omega: Tensor.t;         (* Variance components *)
  gamma: Tensor.t;         (* Random effects *)
  eta: Tensor.t;          (* Linear predictor *)
  y_tilde: Tensor.t;      (* Working response *)
  w: Tensor.t;            (* Working weights *)
  psi: Tensor.t;          (* Objective function value *)
  converged: bool;        (* Convergence flag *)
}

let initialize_state model =
  let eta_init = match model.spec.distribution with
    | Binomial {trials} -> 
        let y_adj = Tensor.add model.y (Tensor.scalar_float 0.5) in
        let m_adj = Tensor.sub trials y_adj |> 
                   Tensor.add (Tensor.scalar_float 0.5) in
        Tensor.log (Tensor.div y_adj m_adj)
    | Poisson -> 
        Tensor.log (Tensor.add model.y (Tensor.scalar_float 0.5))
    | Normal _ -> 
        model.y in

  let w_init = match model.spec.distribution with
    | Binomial {trials} ->
        let p = Tensor.sigmoid eta_init in
        Tensor.mul trials (Tensor.mul p (Tensor.sub (Tensor.ones_like p) p))
    | Poisson ->
        Tensor.exp eta_init
    | Normal {variance} ->
        Tensor.ones [model.spec.n_obs; 1] |>
        Tensor.mul_scalar (Scalar.float (1.0 /. variance)) in

  {
    beta = Tensor.zeros [model.spec.n_fixed; 1];
    omega = Tensor.ones [2; 1];
    gamma = Tensor.zeros [model.spec.n_random; 1];
    eta = eta_init;
    y_tilde = eta_init;
    w = w_init;
    psi = Tensor.zeros [];
    converged = false;
  }

let update_working_values model state =
  match model.spec.distribution with
  | Binomial {trials} ->
      let p = Tensor.sigmoid state.eta in
      let w = Tensor.mul trials (Tensor.mul p (Tensor.sub (Tensor.ones_like p) p)) in
      let y_tilde = Tensor.add state.eta 
        (Tensor.div (Tensor.sub model.y (Tensor.mul trials p)) w) in
      { state with w; y_tilde }
  | Poisson ->
      let mu = Tensor.exp state.eta in
      let w = mu in
      let y_tilde = Tensor.add state.eta 
        (Tensor.div (Tensor.sub model.y mu) w) in
      { state with w; y_tilde }
  | Normal {variance} ->
      { state with 
        w = Tensor.ones_like state.eta |> 
            Tensor.mul_scalar (Scalar.float (1.0 /. variance));
        y_tilde = model.y }

let newton_raphson_update model state =
  let r = Glmm.compute_r model state.w (Tensor.zeros [2; 1]) in
  let r_inv = safe_inverse r in

  let diff = Tensor.sub state.y_tilde 
              (Tensor.add (Tensor.mm model.x state.beta)
                         (Tensor.mm model.z state.gamma)) in
  
  let grad_beta = Tensor.mm (Tensor.transpose model.x ~dim0:0 ~dim1:1)
                           (Tensor.mm r_inv diff) in
  let grad_omega = Tensor.mm (Tensor.transpose model.z ~dim0:0 ~dim1:1)
                            (Tensor.mm r_inv diff) in

  let h_beta = Tensor.mm (Tensor.transpose model.x ~dim0:0 ~dim1:1)
                        (Tensor.mm r_inv model.x) |> Tensor.neg in
  let h_omega = Tensor.mm (Tensor.transpose model.z ~dim0:0 ~dim1:1)
                         (Tensor.mm r_inv model.z) |> Tensor.neg in

  let beta_update = safe_solve h_beta grad_beta |> Tensor.neg in
  let omega_update = safe_solve h_omega grad_omega |> Tensor.neg in

  { state with 
    beta = Tensor.add state.beta beta_update;
    omega = Tensor.add state.omega omega_update }

let check_convergence state prev_state tol =
  let diff_beta = Tensor.sub state.beta prev_state.beta |>
                 Tensor.norm ~p:(Scalar.float 2.0) in
  let diff_omega = Tensor.sub state.omega prev_state.omega |>
                  Tensor.norm ~p:(Scalar.float 2.0) in
  
  let converged = 
    (Tensor.scalar_to_float diff_beta |> Scalar.to_float) < tol &&
    (Tensor.scalar_to_float diff_omega |> Scalar.to_float) < tol in
  
  { state with converged }

let iterate model state =
  let eta = Tensor.add (Tensor.mm model.x state.beta)
                      (Tensor.mm model.z state.gamma) in
  
  let state = { state with eta } in
  let state = update_working_values model state in

  let state = newton_raphson_update model state in

  let gamma = Prediction.predict_random_effects model state in
  { state with gamma }

let run ?(max_iter=100) ?(tol=1e-6) model =
  let rec loop state iter =
    if iter >= max_iter || state.converged then state
    else
      let new_state = iterate model state in
      let new_state = check_convergence new_state state tol in
      let psi = Glmm.compute_psi model 
                  (Tensor.zeros [model.spec.n_fixed; 1])
                  (Tensor.zeros [2; 1])
                  new_state.gamma in
      loop { new_state with psi } (iter + 1)
  in
  
  let initial_state = initialize_state model in
  loop initial_state 0