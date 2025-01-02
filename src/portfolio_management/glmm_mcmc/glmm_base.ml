open Torch

let rmvnorm mean cov =
  let l = Numerical.safe_cholesky cov in
  let z = Tensor.randn [Tensor.size mean 0] in
  Tensor.(add mean (mm l z))

let rgamma shape rate =
  let alpha = Tensor.float_value shape in
  let beta = Tensor.float_value rate in
  let d = alpha -. 1.0 /. 3.0 in
  let c = 1.0 /. Float.sqrt (9.0 *. d) in
  
  let rec loop () =
    let x = Tensor.randn [1] in
    let v = Tensor.float_value (Tensor.add (float_tensor [1.0]) (mul x (float_tensor [c]))) in
    if v <= 0.0 then loop ()
    else
      let v3 = v *. v *. v in
      let u = Tensor.float_value (Tensor.rand [1]) in
      if u < 1.0 -. 0.0331 *. x *. x *. x *. x then
        d *. v3 /. beta
      else if Float.log u < 0.5 *. x *. x +. d *. (1.0 -. v3 +. Float.log v3) then
        d *. v3 /. beta
      else
        loop ()
  in
  Tensor.float_tensor [loop ()]

let linear_predictor x z beta u =
  let x_part = Tensor.mm x beta in
  let z_part = Tensor.mm z u in
  Tensor.add x_part z_part