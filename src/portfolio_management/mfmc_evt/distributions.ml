open Torch
open Types

module type Distribution = sig
  type params
  val sample : params -> int -> (Tensor.t, mf_error) result
  val log_likelihood : params -> Tensor.t -> (Tensor.t, mf_error) result
  val fit_mle : Tensor.t -> (params, mf_error) result
  val params_to_tensor : params -> (Tensor.t, mf_error) result
  val params_from_tensor : Tensor.t -> (params, mf_error) result
end

module BivariateGaussian : Distribution with type params = distribution_params = struct
  type params = distribution_params

  let sample params n =
    match params with
    | BivariateGaussianParams { mu1; mu2; sigma1; sigma2; rho } ->
      let z1 = Tensor.randn [n] in
      let z2 = Tensor.randn [n] in
      let x1 = Tensor.(float mu1 + float sigma1 * z1) in
      let x2 = Tensor.(float mu2 + float sigma2 * (float rho * z1 + float (sqrt (1. -. rho *. rho)) * z2)) in
      Ok (Tensor.stack [x1; x2] ~dim:1)
    | _ -> Error (InvalidParameters "BivariateGaussian.sample: Invalid parameters")

  let log_likelihood params x =
    match params with
    | BivariateGaussianParams { mu1; mu2; sigma1; sigma2; rho } ->
      let x1 = Tensor.select x ~dim:1 ~index:0 in
      let x2 = Tensor.select x ~dim:1 ~index:1 in
      let z1 = Tensor.((x1 - float mu1) / float sigma1) in
      let z2 = Tensor.((x2 - float mu2) / float sigma2) in
      let term1 = Tensor.((z1 * z1 + z2 * z2 - float (2. *. rho) * z1 * z2) / (2. * (float (1. -. rho *. rho)))) in
      let term2 = Tensor.(log (float (2. *. Float.pi *. sigma1 *. sigma2 *. sqrt (1. -. rho *. rho)))) in
      Ok Tensor.(neg (term1 + term2))
    | _ -> Error (InvalidParameters "BivariateGaussian.log_likelihood: Invalid parameters")

  let fit_mle x =
    let x1 = Tensor.select x ~dim:1 ~index:0 in
    let x2 = Tensor.select x ~dim:1 ~index:1 in
    let mu1 = Tensor.mean x1 in
    let mu2 = Tensor.mean x2 in
    let sigma1 = Tensor.std x1 ~unbiased:true in
    let sigma2 = Tensor.std x2 ~unbiased:true in
    let rho = Tensor.corrcoef x1 x2 in
    Ok (BivariateGaussianParams {
      mu1 = Tensor.float_value mu1;
      mu2 = Tensor.float_value mu2;
      sigma1 = Tensor.float_value sigma1;
      sigma2 = Tensor.float_value sigma2;
      rho = Tensor.float_value rho
    })

  let params_to_tensor = function
    | BivariateGaussianParams { mu1; mu2; sigma1; sigma2; rho } ->
      Ok (Tensor.of_float2 [|[|mu1; mu2; sigma1; sigma2; rho|]|])
    | _ -> Error (InvalidParameters "BivariateGaussian.params_to_tensor: Invalid parameters")

  let params_from_tensor t =
    try
      Ok (BivariateGaussianParams {
        mu1 = Tensor.get t 0 0 |> Tensor.float_value;
        mu2 = Tensor.get t 0 1 |> Tensor.float_value;
        sigma1 = Tensor.get t 0 2 |> Tensor.float_value;
        sigma2 = Tensor.get t 0 3 |> Tensor.float_value;
        rho = Tensor.get t 0 4 |> Tensor.float_value
      })
    with _ -> Error (InvalidParameters "BivariateGaussian.params_from_tensor: Invalid tensor")
end

module BivariateGumbel : Distribution with type params = distribution_params = struct
  type params = distribution_params

  let logistic_pickands_function t r =
    Tensor.((t ** (float (1. /. r)) + ((float 1.) - t) ** (float (1. /. r))) ** (float r))

  let sample params n =
    match params with
    | BivariateGumbelParams { mu1; mu2; beta1; beta2; r } ->
      let open Tensor in
      let u = uniform ~from:0. ~to_:1. [n] in
      let v = uniform ~from:0. ~to_:1. [n] in
      let t = u in
      let w = neg (log (neg (log v))) in
      let z = logistic_pickands_function t (float r) in
      let x1 = float mu1 - float beta1 * log (neg (log (t * w / z))) in
      let x2 = float mu2 - float beta2 * log (neg (log ((float 1. - t) * w / z))) in
      Ok (stack [x1; x2] ~dim:1)
    | _ -> Error (InvalidParameters "BivariateGumbel.sample: Invalid parameters")

  let log_likelihood params x =
    match params with
    | BivariateGumbelParams { mu1; mu2; beta1; beta2; r } ->
      let open Tensor in
      let x1 = select x ~dim:1 ~index:0 in
      let x2 = select x ~dim:1 ~index:1 in
      let z1 = (x1 - float mu1) / float beta1 in
      let z2 = (x2 - float mu2) / float beta2 in
      let u = exp (neg z1) in
      let v = exp (neg z2) in
      let t = u / (u + v) in
      let w = u + v in
      let a = logistic_pickands_function t (float r) in
      Ok ((float (-1. /. r) -. 1.) * log a + log (float r) - log (float beta1) - log (float beta2) - 
          z1 - z2 - u - v - w * a)
    | _ -> Error (InvalidParameters "BivariateGumbel.log_likelihood: Invalid parameters")

  let fit_mle x =
    (* Implement MLE for BivariateGumbel *)
    Error (FittingError "BivariateGumbel.fit_mle: Not implemented")

  let params_to_tensor = function
    | BivariateGumbelParams { mu1; mu2; beta1; beta2; r } ->
      Ok (Tensor.of_float2 [|[|mu1; mu2; beta1; beta2; r|]|])
    | _ -> Error (InvalidParameters "BivariateGumbel.params_to_tensor: Invalid parameters")

  let params_from_tensor t =
    try
      Ok (BivariateGumbelParams { 
        mu1 = Tensor.get t 0 0 |> Tensor.float_value;
        mu2 = Tensor.get t 0 1 |> Tensor.float_value;
        beta1 = Tensor.get t 0 2 |> Tensor.float_value;
        beta2 = Tensor.get t 0 3 |> Tensor.float_value;
        r = Tensor.get t 0 4 |> Tensor.float_value
      })
    with _ -> Error (InvalidParameters "BivariateGumbel.params_from_tensor: Invalid tensor")
end

module Bernoulli : Distribution with type params = distribution_params = struct
  type params = distribution_params

  let sample params n =
    match params with
    | GaussianParams { mu = p; sigma = _ } ->
      Ok (Tensor.bernoulli ~p:(Tensor.float p) [n])
    | _ -> Error (InvalidParameters "Bernoulli.sample: Invalid parameters")

  let log_likelihood params x =
    match params with
    | GaussianParams { mu = p; sigma = _ } ->
      Ok Tensor.(x * log (float p) + (float 1. - x) * log (float (1. -. p)))
    | _ -> Error (InvalidParameters "Bernoulli.log_likelihood: Invalid parameters")

  let fit_mle x =
    let p = Tensor.mean x in
    Ok (GaussianParams { mu = Tensor.float_value p; sigma = 0. })

  let params_to_tensor = function
    | GaussianParams { mu = p; sigma = _ } -> Ok (Tensor.of_float1 [|p|])
    | _ -> Error (InvalidParameters "Bernoulli.params_to_tensor: Invalid parameters")

  let params_from_tensor t =
    try
      Ok (GaussianParams { mu = Tensor.get t 0 |> Tensor.float_value; sigma = 0. })
    with _ -> Error (InvalidParameters "Bernoulli.params_from_tensor: Invalid tensor")
end