open Torch
open Types

let solve_eqg params ode_solution T =
  let n_steps = 1000 in
  let dt = T /. float_of_int n_steps in
  
  let phi = Tensor.zeros [n_steps + 1] in
  let psi = Tensor.zeros [n_steps + 1] in
  let chi = Tensor.zeros [n_steps + 1] in
  
  Tensor.set phi [|n_steps|] (Scalar.float 0.);
  Tensor.set psi [|n_steps|] (Scalar.float 0.);
  Tensor.set chi [|n_steps|] (Scalar.float 0.);
  
  for i = n_steps - 1 downto 0 do
    let t = float_of_int i *. dt in
    let phi_t = Tensor.get phi [|i + 1|] in
    let psi_t = Tensor.get psi [|i + 1|] in
    let chi_t = Tensor.get chi [|i + 1|] in
    
    let a00_t = Tensor.get ode_solution.a00 [|i|] in
    let a11_t = Tensor.get ode_solution.a11 [|i|] in
    let a10_t = Tensor.get ode_solution.a10 [|i|] in
    let b0_t = Tensor.get ode_solution.b0 [|i|] in
    let b1_t = Tensor.get ode_solution.b1 [|i|] in
    
    let dphi = Tensor.(
      sub (mul (float (-0.5 *. params.gamma)) (mul a00_t (mul (float params.sigma0) (float params.sigma0))))
          (add (mul (float 0.5) (mul phi_t phi_t))
               (mul (float params.k0) phi_t))
    ) in
    
    let dpsi = Tensor.(
      sub (mul (float (-0.5 *. params.gamma)) (mul a11_t (mul (float params.sigma) (float params.sigma))))
          (add (mul (float 0.5) (mul psi_t psi_t))
               (mul (float params.k) psi_t))
    ) in
    
    let dchi = Tensor.(
      sub (mul (float (-1. *. params.gamma)) (mul a10_t (mul (float params.sigma0) (float params.sigma))))
          (add (mul phi_t psi_t)
               (mul (float (params.k0 +. params.k)) chi_t))
    ) in
    
    Tensor.set phi [|i|] Tensor.(sub phi_t (mul (float dt) dphi));
    Tensor.set psi [|i|] Tensor.(sub psi_t (mul (float dt) dpsi));
    Tensor.set chi [|i|] Tensor.(sub chi_t (mul (float dt) dchi));
  done;
  
  { phi; psi; chi }

let calculate_value_function eqg_solution x0 xi =
  let { phi; psi; chi } = eqg_solution in
  Tensor.(
    add (add (mul (float 0.5) (mul (mul (transpose x0) phi) x0))
             (mul (float 0.5) (mul (mul (transpose xi) psi) xi)))
        (mul (mul (transpose x0) chi) xi)
  )