type parameters = {
  sigma: float;
  alpha: float;
  mu: float;
  r: float;
}

type classical_path = {
  x: float -> float;
  p: float -> float;
}

type feller_state = {
  x: float;
  mu_x: float;
  sigma_x: float;
  a: float;
  b: float;
  jacobian: float;
}

let pi = Torch.Scalar.pi

let erf x =
  let rec series_sum n acc term x2 =
    if n >= 20 || abs_float term < 1e-15 then acc
    else
      let new_term = term *. (-. x2) *. float_of_int(2 * n - 1) /. 
                    (float_of_int n *. float_of_int(2 * n + 1)) in
      series_sum (n + 1) (acc +. new_term) new_term x2
  in
  if x = 0.0 then 0.0
  else
    let x2 = x *. x in
    2.0 *. series_sum 1 x x x2 /. sqrt pi

let normal_cdf x =
  0.5 *. (1.0 +. erf (x /. sqrt 2.0))

let rec integrate_adaptive f a b tol =
  let h = b -. a in
  let fa = f a and fb = f b in
  let fd = f (a +. h /. 2.0) in
  let s1 = h *. (fa +. fb) /. 2.0 in
  let s2 = h *. (fa +. 2.0 *. fd +. fb) /. 4.0 in
  if abs_float (s2 -. s1) <= 3.0 *. tol then
    s2 +. (s2 -. s1) /. 15.0
  else
    let mid = (a +. b) /. 2.0 in
    integrate_adaptive f a mid (tol /. 2.0) +.
    integrate_adaptive f mid b (tol /. 2.0)

let solve_ode f x0 t0 t1 dt =
  let rec integrate x t =
    if t >= t1 then x
    else
      let k1 = f t x in
      let k2 = f (t +. dt/.2.0) (x +. dt*.k1/.2.0) in
      let k3 = f (t +. dt/.2.0) (x +. dt*.k2/.2.0) in
      let k4 = f (t +. dt) (x +. dt*.k3) in
      integrate (x +. dt*.(k1 +. 2.0*.k2 +. 2.0*.k3 +. k4)/.6.0) (t +. dt)
  in
  integrate x0 t0

let integrate f a b n =
  let h = (b -. a) /. float_of_int n in
  let sum = ref ((f a +. f b) /. 2.0) in
  for i = 1 to n - 1 do
    sum := !sum +. f (a +. float_of_int i *. h)
  done;
  !sum *. h

let to_feller params s =
  if params.alpha >= 0.0 then
    invalid_arg "Alpha must be negative for CEV model";
  let x = Float.pow s (-.2.0 *. params.alpha) /. 
          (params.sigma *. params.sigma *. params.alpha *. params.alpha) in
  let a = 2.0 +. 1.0 /. params.alpha in
  let b = 2.0 *. params.alpha *. params.mu in
  let dx_ds = -.2.0 *. params.alpha *. 
              Float.pow s (-.2.0 *. params.alpha -. 1.0) /. 
              (params.sigma *. params.sigma *. params.alpha *. params.alpha) in
  {Types.
    x;
    mu_x = params.mu *. s *. dx_ds;
    sigma_x = params.sigma *. Float.pow s (params.alpha +. 1.0) *. dx_ds;
    a;
    b;
    jacobian = abs_float dx_ds;
  }

let from_feller params x =
  Float.pow (params.sigma *. params.sigma *. params.alpha *. params.alpha *. x)
            (-1.0 /. (2.0 *. params.alpha))

let verify_transform params s =
  let f = to_feller params s in
  abs_float (from_feller params f.x -. s) < 1e-10

let find_classical_path params x_t x_T t =
  let sigma2 = params.sigma *. params.sigma in
  let mu = params.r -. sigma2 /. 2.0 in
  let p0 = (x_T -. x_t +. mu *. t) /. (sigma2 *. t) in
  {Types.
    x = (fun tau -> x_t +. (sigma2 *. p0 -. mu) *. tau);
    p = (fun _ -> p0);
  }

let compute_action params x_t x_T t =
  let path = find_classical_path params x_t x_T t in
  let p0 = path.p 0.0 in
  let sigma2 = params.sigma *. params.sigma in
  0.5 *. sigma2 *. t *. p0 *. p0 +. params.r *. t

let kernel params x_t x_T t =
  let action = compute_action params x_t x_T t in
  1.0 /. sqrt (2.0 *. Float.pi *. params.sigma *. params.sigma *. t) *.
  exp (-. action)

let d1 s k r sigma t =
  (log (s /. k) +. (r +. sigma *. sigma /. 2.0) *. t) /. 
  (sigma *. sqrt t)

let d2 s k r sigma t =
  d1 s k r sigma t -. sigma *. sqrt t

let call_price params s k t =
  let d1 = d1 s k params.r params.sigma t in
  let d2 = d2 s k params.r params.sigma t in
  s *. Math.normal_cdf d1 -. 
  k *. exp (-.params.r *. t) *. Math.normal_cdf d2

let call_delta params s k t =
  let d1 = d1 s k params.r params.sigma t in
  Math.normal_cdf d1

let call_gamma params s k t =
  let d1 = d1 s k params.r params.sigma t in
  exp (-.d1 *. d1 /. 2.0) /. 
  (s *. params.sigma *. sqrt (2.0 *. Float.pi *. t))

let validate_parameters params =
  if params.sigma <= 0.0 then
    Error "Volatility must be positive"
  else if params.alpha >= 0.0 then
    Error "Alpha must be negative for CEV model"
  else
    Ok params

let european_call params s k t =
  match validate_parameters params with
  | Error msg -> invalid_arg msg
  | Ok params ->
      let f = Transform.to_feller params s in
      let integrand s_T =
        let prop = HeatKernel.kernel params s s_T t in
        let payoff = max 0.0 (s_T -. k) in
        prop *. payoff
      in
      let price = Math.integrate_adaptive 
                   integrand k (s *. 5.0) 1e-6 in
      exp (-.params.r *. t) *. price