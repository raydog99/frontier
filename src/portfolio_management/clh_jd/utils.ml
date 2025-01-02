open Torch

let normal_cdf x =
  let open Tensor in
  let sqrt_2 = sqrt 2. in
  (erf (x / sqrt_2) + float 1.) / float 2.

let normal_pdf x =
  let open Tensor in
  exp (-(x ** float 2.) / float 2.) / sqrt (float 2. *. Stdlib.acos (-1.))