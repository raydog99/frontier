open Torch

let compute_spectral_norm matrix =
  let n = Tensor.size matrix 0 in
  let v = Tensor.randn [|n|] in
  let v = Tensor.div v (Tensor.norm v) in
  
  let rec power_iterate v iter =
    if iter >= 20 then v
    else
      let mv = Tensor.mm matrix v in
      let mv_norm = Tensor.norm mv in
      let v_next = Tensor.div mv mv_norm in
      power_iterate v_next (iter + 1) in
  
  let final_v = power_iterate v 0 in
  let rayleigh = Tensor.dot (Tensor.mm matrix final_v) final_v in
  Tensor.float_value rayleigh

let approximate matrix epsilon =
  let spec_norm = compute_spectral_norm matrix in
  let degree = Int.of_float (
    max (spec_norm /. epsilon)
        (Float.log (2.0 /. epsilon))
  ) in
  
  let identity = Tensor.eye (Tensor.size matrix 0) in
  let term = ref identity in
  let sum = ref identity in
  let factorial = ref 1.0 in
  
  for i = 1 to degree do
    term := Tensor.mm !term matrix;
    factorial := !factorial *. Float.of_int i;
    let scaled_term = Tensor.div_scalar !term !factorial in
    sum := Tensor.add !sum scaled_term
  done;
  !sum