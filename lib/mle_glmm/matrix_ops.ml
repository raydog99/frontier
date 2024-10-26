open Torch

let safe_solve a b =
  let n = Tensor.size a 0 in
  let reg_strength = ref 1e-10 in
  let max_attempts = 10 in
  
  let rec attempt count =
    if count >= max_attempts then
      failwith "Matrix solve failed after maximum attempts"
    else
      try
        let reg = Tensor.eye n |> 
                  Tensor.mul_scalar (Scalar.float !reg_strength) in
        let a' = Tensor.add a reg in
        Tensor.solve a' b
      with _ ->
        reg_strength := !reg_strength *. 10.0;
        attempt (count + 1)
  in
  attempt 0

let safe_cholesky mat =
  let n = Tensor.size mat 0 in
  let jitter = ref 1e-10 in
  let max_attempts = 10 in
  
  let rec attempt count =
    if count >= max_attempts then
      failwith "Cholesky decomposition failed after maximum attempts"
    else
      try
        let reg = Tensor.eye n |> 
                  Tensor.mul_scalar (Scalar.float !jitter) in
        let mat' = Tensor.add mat reg in
        Tensor.cholesky mat' Upper
      with _ ->
        jitter := !jitter *. 10.0;
        attempt (count + 1)
  in
  attempt 0

let bessel_k nu x =
  match Float.to_int (Scalar.to_float nu) with
  | 0 -> Tensor.exp (Tensor.neg x)
  | 1 -> 
      let k1_approx = Tensor.exp (Tensor.neg x) |> 
                     Tensor.mul (Tensor.add (Tensor.ones_like x) 
                                          (Tensor.div (Tensor.ones_like x) x)) in
      Tensor.sqrt (Tensor.div (Tensor.mul_scalar Tensor.pi (Scalar.float 2.0)) x) 
      |> Tensor.mul k1_approx
  | n when n > 1 ->
      let kn_approx = Tensor.exp (Tensor.neg x) |>
                     Tensor.mul (Tensor.sqrt (Tensor.div (Tensor.mul_scalar Tensor.pi 
                                                       (Scalar.float 2.0)) x)) in
      Tensor.mul kn_approx (Tensor.pow x (Scalar.float (float_of_int (-n))))
  | _ -> failwith "Invalid order for Bessel function"

let log_det mat =
  let chol = safe_cholesky mat in
  let diag = Tensor.diagonal chol ~dim1:0 ~dim2:1 in
  Tensor.log diag |> Tensor.sum |> Tensor.mul_scalar (Scalar.float 2.0)

let moore_penrose_inverse mat =
  let (u, s, v) = Tensor.svd mat ~some:false in
  let s_inv = Tensor.reciprocal s |>
              Tensor.where_ ~condition:(Tensor.gt s (Tensor.scalar_float 1e-10)) 
                          ~other:(Tensor.zeros_like s) in
  let s_inv_mat = Tensor.diag s_inv in
  Tensor.mm (Tensor.mm v s_inv_mat) (Tensor.transpose u ~dim0:0 ~dim1:1)

let standard_errors info =
  Tensor.sqrt (Tensor.diagonal (safe_solve info (Tensor.eye (Tensor.size info 0))) 
              ~dim1:0 ~dim2:1)