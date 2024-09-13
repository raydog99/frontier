open Torch

type t = {
  hurst: float;
  eta: float;
  kappa: float;
}

let create ~hurst ~eta ~kappa = { hurst; eta; kappa }

let sample t n =
  let h = t.hurst in
  let eta = t.eta in
  let kappa = t.kappa in
  let dt = 1. /. float_of_int n in
  let dw = Tensor.randn [n] in
  let ou = Tensor.zeros [n] in
  for i = 1 to n - 1 do
    let prev = Tensor.get ou (i - 1) in
    let drift = Tensor.((neg (mul_scalar prev kappa)) |> mul_scalar dt) in
    let diffusion = Tensor.(mul_scalar dw.(i) (sqrt (mul_scalar dt (2. *. eta *. eta)))) in
    Tensor.set ou i Tensor.(add prev (add drift diffusion))
  done;
  ou

let autocorrelation t s =
  let h = t.hurst in
  let kappa = t.kappa in
  (2. *. sin (Float.pi *. h) /. Float.pi) *.
  (let x = Tensor.arange ~start:0. ~end_:10. ~step:0.01 ~options:(Kind K.Float, Device.Cpu) in
   let integrand = Tensor.(
     cos (mul_scalar x s) * 
     pow x (h -. 0.5) /
     (pow x 2. + 1.)
   ) in
   Tensor.sum integrand |> Tensor.to_float0_exn) *. 0.01

let variance t =
  let h = t.hurst in
  let eta = t.eta in
  let kappa = t.kappa in
  (eta ** 2. *. (2. *. h +. 1.)) /. (2. *. (kappa ** (2. *. h)))

let conditional_probability t x m =
  let rho = autocorrelation t m in
  let var = variance t in
  let z = (rho *. (x -. 0.5)) /. (sqrt var *. sqrt (1. -. rho ** 2.)) in
  Stats.normal_cdf z

let minimum_autocorrelation t =
  let f s = -.autocorrelation t s in
  let s_opt = Utils.find_root (fun s -> (autocorrelation t s) +. 1.) 0. 10. 1e-6 in
  (s_opt, f s_opt)

let optimal_lag t =
  let h = t.hurst in
  let _, s_star = minimum_autocorrelation t in
  s_star *. (t.kappa ** (1. -. 2. *. h))