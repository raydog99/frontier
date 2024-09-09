open Torch

type t = Tensor.t -> Tensor.t -> Tensor.t

let squared_error z y =
  Tensor.pow (Tensor.sub z y) (Tensor.of_float 2.)

let pinball_loss alpha z y =
  let diff = Tensor.sub y z in
  let pos = Tensor.relu diff in
  let neg = Tensor.relu (Tensor.neg diff) in
  Tensor.add (Tensor.mul (Tensor.of_float alpha) pos)
             (Tensor.mul (Tensor.of_float (1. -. alpha)) neg)

let expectile_loss tau z y =
  let diff = Tensor.sub y z in
  let weight = Tensor.where_ (Tensor.gt diff (Tensor.of_float 0.))
                (Tensor.of_float tau)
                (Tensor.of_float (1. -. tau)) in
  Tensor.mul weight (Tensor.pow diff (Tensor.of_float 2.))

let var_score alpha z y =
  pinball_loss alpha z y

let es_score alpha z1 z2 y =
  let var_part = var_score alpha z1 y in
  let es_part = Tensor.relu (Tensor.sub y z1) in
  Tensor.add var_part (Tensor.mul (Tensor.of_float (1. /. (1. -. alpha))) es_part)

let b_homogeneous_mean b z y =
  match b with
  | b when b = 0. -> 
      let y_div_z = Tensor.div y z in
      Tensor.sub (Tensor.div y z) (Tensor.log y_div_z)
  | b when b = 1. ->
      let y_div_z = Tensor.div y z in
      Tensor.sub (Tensor.mul y (Tensor.log y_div_z)) (Tensor.sub y z)
  | _ ->
      let term1 = Tensor.div (Tensor.pow y (Tensor.of_float b)) (Tensor.of_float (b *. (b -. 1.))) in
      let term2 = Tensor.div (Tensor.pow z (Tensor.of_float (b -. 1.))) (Tensor.of_float (b -. 1.)) in
      let term3 = Tensor.mul (Tensor.pow z (Tensor.of_float b)) (Tensor.of_float b) in
      Tensor.sub (Tensor.sub term1 (Tensor.mul y term2)) term3