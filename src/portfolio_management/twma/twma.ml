open Torch

type t = {
  mutable weights: Tensor.t;
  mutable ma: Tensor.t;
  window_size: int;
  alpha: float;
}

let create window_size alpha =
  {
    weights = Tensor.ones [window_size];
    ma = Tensor.zeros [1];
    window_size;
    alpha;
  }

let update t price =
  let price_tensor = Tensor.of_float1 [|price|] in
  let new_ma = Tensor.(mul t.weights price_tensor |> sum) in
  t.ma <- Tensor.(mul_scalar t.ma (1. -. t.alpha) + mul_scalar new_ma t.alpha);
  t.weights <- Tensor.(roll t.weights ~shifts:(-1) ~dims:[0]);
  Tensor.(copy_ (select t.weights 0 (-1)) price_tensor)

let get_ma t =
  Tensor.to_float0_exn t.ma