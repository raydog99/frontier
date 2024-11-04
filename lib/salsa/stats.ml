open Torch

let compute_acf series max_lag =
  let n = Tensor.shape1_exn series in
  let mean = Tensor.mean series in
  let centered = Tensor.sub series mean in
  let variance = Tensor.mean (Tensor.mul centered centered) in
  
  let acf = Tensor.zeros [max_lag + 1] in
  for k = 0 to max_lag do
    let shifted = Tensor.narrow series ~dim:0 ~start:k ~length:(n - k) in
    let orig = Tensor.narrow series ~dim:0 ~start:0 ~length:(n - k) in
    let cov = Tensor.mean (Tensor.mul 
      (Tensor.sub shifted mean) 
      (Tensor.sub orig mean)) in
    Tensor.copy_ 
      (Tensor.narrow acf ~dim:0 ~start:k ~length:1) 
      (Tensor.div_scalar cov (Tensor.item variance))
  done;
  acf

let ljung_box_test residuals max_lag =
  let n = float_of_int (Tensor.shape1_exn residuals) in
  let acf = compute_acf residuals max_lag in
  let stat = Tensor.zeros [1] in
  
  for k = 1 to max_lag do
    let rk = Tensor.get acf [k] in
    let term = (rk *. rk) /. (n -. float_of_int k) in
    Tensor.add_ stat (Tensor.full [1] (term *. n *. (n +. 2.)))
  done;
  Tensor.item stat

let durbin_watson residuals =
  let n = Tensor.shape1_exn residuals in
  let diff = Tensor.sub 
    (Tensor.narrow residuals ~dim:0 ~start:1 ~length:(n-1))
    (Tensor.narrow residuals ~dim:0 ~start:0 ~length:(n-1)) in
  let num = Tensor.sum (Tensor.mul diff diff) in
  let den = Tensor.sum (Tensor.mul residuals residuals) in
  Tensor.item (Tensor.div num den)