open Torch

let mean data = List.fold_left (+.) 0. data /. float_of_int (List.length data)

let variance data =
  let m = mean data in
  let sum_sq_diff = List.fold_left (fun acc x -> acc +. (x -. m) ** 2.) 0. data in
  sum_sq_diff /. float_of_int (List.length data - 1)

let standard_error data = sqrt (variance data /. float_of_int (List.length data))

let confidence_interval data alpha =
  let n = float_of_int (List.length data) in
  let se = standard_error data in
  let t_score = Tensor.distributions.StudentT.inv_cdf (Tensor.of_float0 ((1. -. alpha) /. 2.)) (n -. 1.) in
  let margin = Tensor.to_float0_exn (Tensor.mul t_score (Tensor.of_float0 se)) in
  let m = mean data in
  (m -. margin, m +. margin)

let t_test data1 data2 =
  let n1 = float_of_int (List.length data1) in
  let n2 = float_of_int (List.length data2) in
  let m1 = mean data1 in
  let m2 = mean data2 in
  let v1 = variance data1 in
  let v2 = variance data2 in
  let se = sqrt (v1 /. n1 +. v2 /. n2) in
  let t_stat = (m1 -. m2) /. se in
  let df = (v1 /. n1 +. v2 /. n2) ** 2. /. ((v1 /. n1) ** 2. /. (n1 -. 1.) +. (v2 /. n2) ** 2. /. (n2 -. 1.)) in
  let p_value = 2. *. (1. -. Tensor.distributions.StudentT.cdf (Tensor.of_float0 (abs_float t_stat)) df) in
  Tensor.to_float0_exn p_value