open Torch

type model_params = {
  alpha: float;
  sigma: float;
  theta: float;
  a: float;
  b: float;
  rho: float;
  t: float;
}

let create_model_params ~alpha ~sigma ~theta ~a ~b ~rho ~t =
  {alpha; sigma; theta; a; b; rho; t}

let create_incomplete_model_params ~alpha ~sigma ~theta ~a ~b ~rho ~t =
  if abs rho >= 1.0 then
    failwith "Incomplete market requires |rho| < 1"
  else
    create_model_params ~alpha ~sigma ~theta ~a ~b ~rho ~t

let wealth_process params initial_wealth control_process t_steps =
  let dt = params.t /. float_of_int t_steps in
  let sqrt_dt = sqrt dt in
  
  let rec process t wealth acc =
    if t >= params.t then List.rev acc
    else
      let dw = Tensor.randn [1] in
      let control = control_process t wealth in
      let drift = Tensor.((params.alpha * wealth) + (control * wealth * params.theta * params.sigma)) in
      let diffusion = Tensor.(control * wealth * params.sigma * dw) in
      let new_wealth = Tensor.(wealth + (drift * float_vec [|dt|]) + (diffusion * float_vec [|sqrt_dt|])) in
      process (t +. dt) new_wealth (new_wealth :: acc)
  in
  
  process 0. initial_wealth [initial_wealth]

let simulate_reference params initial_reference t_steps =
  let dt = params.t /. float_of_int t_steps in
  let sqrt_dt = sqrt dt in
  
  let rec simulate t reference acc =
    if t >= params.t then List.rev acc
    else
      let dw = Tensor.randn [1] in
      let drift = Tensor.(params.a * reference) in
      let diffusion = Tensor.(params.b * reference * dw) in
      let new_reference = Tensor.(reference + (drift * float_vec [|dt|]) + (diffusion * float_vec [|sqrt_dt|])) in
      simulate (t +. dt) new_reference (new_reference :: acc)
  in
  
  simulate 0. initial_reference [initial_reference]