open Torch

type t = {
  hurst: float;
  scale: float;
}

let create ~hurst ~scale = { hurst; scale }

let sample t n =
  let h = t.hurst in
  let c = t.scale in
  let t = Tensor.arange ~start:1 ~end_:(n + 1) ~options:(Kind K.Float, Device.Cpu) in
  let cov = Tensor.(
    pow_scalar (abs (sub (unsqueeze t ~dim:1) (unsqueeze t ~dim:0))) (2. *. h) 
    |> mul_scalar (c *. c)
  ) in
  Tensor.multivariate_normal (Tensor.zeros [n]) cov

let increment_variance t s =
  t.scale ** 2. *. (s ** (2. *. t.hurst))