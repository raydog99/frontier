open Torch

let safe_cholesky input =
  let jitter = Tensor.eye (Tensor.size input 0) in
  let rec attempt scale =
    try
      Tensor.cholesky (Tensor.add input (Tensor.mul jitter (Tensor.float_tensor [scale])))
    with _ ->
      if scale > 1e-3 then
        failwith "Cholesky decomposition failed"
      else
        attempt (scale *. 10.)
  in
  attempt 1e-10

let safe_inverse input =
  let n = Tensor.size input 0 in
  try
    Tensor.inverse input
  with _ ->
    let l = safe_cholesky input in
    let eye = Tensor.eye n in
    Tensor.(mm (transpose l) (mm (inverse l) eye))

let log_sum_exp x =
  let max_x = Tensor.max x in
  let shifted = Tensor.sub x max_x in
  Tensor.(add max_x (log (sum (exp shifted))))

let stable_sigmoid x =
  let neg_x = Tensor.neg x in
  Tensor.(
    where (gt x (float_tensor [0.]))
      (div (float_tensor [1.]) (add (float_tensor [1.]) (exp neg_x)))
      (let exp_x = exp x in
       div exp_x (add (float_tensor [1.]) exp_x))
  )

let log1p_exp x =
  Tensor.(
    where (gt x (float_tensor [35.]))
      x
      (where (lt x (float_tensor [-10.]))
        (exp x)
        (log1p (exp x)))
  )