open Torch

let arctan_transform x =
  Tensor.(scalar 0.5 + scalar (1. /. Float.pi) * atan (sub x (scalar 0.5)))

let inverse_arctan_transform x =
  Tensor.(scalar 0.5 + tan (mul (sub x (scalar 0.5)) (scalar Float.pi)))

let integrate f a b n =
  let dx = (b -. a) /. float_of_int n in
  let sum = ref 0. in
  for i = 0 to n - 1 do
    let x = a +. dx *. (float_of_int i +. 0.5) in
    sum := !sum +. f x
  done;
  !sum *. dx

let linspace start stop num =
  Tensor.linspace ~start ~end_:stop ~steps:num ~options:(Kind K.Float, Device.Cpu)

let rec find_root f a b tol =
  let c = (a +. b) /. 2. in
  let fc = f c in
  if abs_float fc < tol then c
  else if fc *. f a > 0. then find_root f c b tol
  else find_root f a c tol

let confidence_interval tensor confidence_level =
  let sorted = Tensor.sort tensor ~dim:0 ~descending:false in
  let n = Tensor.shape sorted |> List.hd in
  let alpha = 1. -. confidence_level in
  let lower_index = int_of_float (float_of_int n *. (alpha /. 2.)) in
  let upper_index = int_of_float (float_of_int n *. (1. -. alpha /. 2.)) in
  (Tensor.get sorted lower_index |> Tensor.to_float0_exn,
   Tensor.get sorted upper_index |> Tensor.to_float0_exn)