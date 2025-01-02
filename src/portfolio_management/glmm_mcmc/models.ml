open Torch
open Base

module Probit = struct
  let link eta = 
    (* Normal CDF approximation *)
    let x = Tensor.div eta (float_tensor [Float.sqrt 2.0]) in
    Tensor.(mul 
      (float_tensor [0.5])
      (add (float_tensor [1.0]) (erf x)))

  let inverse_link p =
    (* Normal quantile function approximation *)
    let p = Tensor.clamp p ~min:1e-7 ~max:(1.0 -. 1e-7) in
    Tensor.(mul 
      (float_tensor [Float.sqrt 2.0])
      (erfinv (mul (sub (mul p (float_tensor [2.0])) (float_tensor [1.0])) 
                  (float_tensor [1.0]))))

  let log_likelihood data params =
    let eta = linear_predictor data.x data.z params.beta params.u in
    let p = link eta in
    let ll = Tensor.(
      add
        (mul data.y (log p))
        (mul (sub (float_tensor [1.0]) data.y) (log (sub (float_tensor [1.0]) p)))
    ) in
    Tensor.float_value (Tensor.sum ll)
end

module Logistic = struct
  let link eta = Numerical.stable_sigmoid eta

  let inverse_link p =
    let p = Tensor.clamp p ~min:1e-7 ~max:(1.0 -. 1e-7) in
    Tensor.(log (div p (sub (float_tensor [1.0]) p)))

  let log_likelihood data params =
    let eta = linear_predictor data.x data.z params.beta params.u in
    let bce = Tensor.binary_cross_entropy_with_logits eta data.y in
    Tensor.float_value (Tensor.neg (Tensor.sum bce))
end

module Poisson = struct
  let link eta = Tensor.exp eta

  let inverse_link mu =
    Tensor.log (Tensor.clamp mu ~min:1e-7)

  let log_likelihood data params =
    let eta = linear_predictor data.x data.z params.beta params.u in
    let mu = link eta in
    let ll = Tensor.(
      sub
        (mul data.y (log mu))
        (add mu (log_factorial data.y))
    ) in
    Tensor.float_value (Tensor.sum ll)
end