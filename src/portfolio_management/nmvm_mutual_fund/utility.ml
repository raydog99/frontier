open Torch

type t = Tensor.t -> Tensor.t

let exponential a w =
  if a <= 0. then failwith "Parameter 'a' must be positive";
  Tensor.(neg (exp (neg (mul_scalar w a))))

let power gamma w =
  if gamma = 0. then failwith "Parameter 'gamma' must not be zero";
  Tensor.(pow w (Scalar.f gamma) / Scalar.f gamma)

let log w =
  Tensor.log w

let quadratic a b w =
  if b >= 0. then failwith "Parameter 'b' must be negative for risk aversion";
  Tensor.(a * w + b * (pow w (Scalar.f 2.0)))

let sahara a b delta w =
  if a <= 0. || b <= 0. then failwith "Parameters 'a' and 'b' must be positive";
  let t = Tensor.(sqrt (pow (Scalar.f b) (Scalar.f 2.0) + pow (sub w (Scalar.f delta)) (Scalar.f 2.0))) in
  Tensor.(neg (div (Scalar.f 1.0) (sub (Scalar.f a) (Scalar.f 1.0))) *
          (sub w (Scalar.f delta) + mul (Scalar.f a) t) /
          pow (add (sub w (Scalar.f delta)) t) (Scalar.f a))

let crra gamma w =
  if gamma = 1.0 then Tensor.log w
  else Tensor.((pow w (Scalar.f (1. -. gamma)) - Scalar.f 1.) / Scalar.f (1. -. gamma))

let cara a w =
  if a <= 0. then failwith "Parameter 'a' must be positive";
  Tensor.(neg (exp (neg (mul_scalar w a))) / Scalar.f a)

let expected_utility utility wealth_distribution =
  Tensor.mean (Tensor.map utility wealth_distribution)