open Torch

type t = {
  mutable h2: Tensor.t;
  mutable h1: Tensor.t;
  mutable h0: Tensor.t;
}

let create () =
  {
    h2 = Tensor.zeros [];
    h1 = Tensor.zeros [];
    h0 = Tensor.zeros [];
  }

let update vf params dt model_type =
  let open Params in
  match model_type with
  | `Model1 ->
      let a1 = params.r_xx -. (params.gamma +. params.r_xa) ** 2. /. (params.s +. params.r_aa) in
      let b1 = (params.gamma +. params.r_xa) *. params.s /. (params.s +. params.r_aa) in
      let c = params.s /. (params.s +. params.r_aa) in

      let dh2 = Tensor.(-(vf.h2 * vf.h2 / (of_float 2. * of_float params.eta) + of_float a1)) in
      let dh1 = Tensor.(-(vf.h1 * vf.h2 / (of_float 2. * of_float params.eta) + of_float b1)) in
      let dh0 = Tensor.(-(
        params.sigma_x ** 2. / 2. * vf.h2 +
        vf.h1 * vf.h1 / (of_float 4. * of_float params.eta) +
        of_float (params.sigma_x ** 2. / 2.) +
        of_float (params.rho * params.sigma_s * params.sigma_x) -
        of_float (0.5 * log c / params.kappa) -
        of_float (params.s / (2. * params.kappa) * (c -. 1.))
      )) in

      vf.h2 <- Tensor.(vf.h2 + dh2 * of_float dt);
      vf.h1 <- Tensor.(vf.h1 + dh1 * of_float dt);
      vf.h0 <- Tensor.(vf.h0 + dh0 * of_float dt)
  | `Model2 ->
      ()

let initialize vf params model_type =
  match model_type with
  | `Model1 -> vf.h2 <- Tensor.of_float 2. *. params.Params.eta
  | `Model2 ->
      let eta_tilde = params.Params.eta -. params.Params.r_vv /. 2. +. params.Params.r_va ** 2. /. (2. *. (params.Params.s +. params.Params.r_aa)) in
      vf.h2 <- Tensor.of_float 2. *. eta_tilde;
  vf.h1 <- Tensor.zeros [];
  vf.h0 <- Tensor.zeros []

let get_h2 vf = Tensor.to_float0_exn vf.h2
let get_h1 vf = Tensor.to_float0_exn vf.h1
let get_h0 vf = Tensor.to_float0_exn vf.h0