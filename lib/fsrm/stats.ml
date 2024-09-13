open Torch

let acf x lag =
  let n = Tensor.shape x |> List.hd in
  let mean = Tensor.mean x in
  let x_centered = Tensor.sub x mean in
  let denominator = Tensor.sum (Tensor.pow x_centered 2.) in
  let r = Tensor.zeros [lag + 1] in
  for k = 0 to lag do
    let numerator = Tensor.sum (Tensor.mul (Tensor.slice x_centered ~dim:0 ~start:0 ~end:(n-k) ())
                                           (Tensor.slice x_centered ~dim:0 ~start:k ~end:n ())) in
    Tensor.set r k (Tensor.div numerator denominator)
  done;
  r

let shannon_entropy x =
  let p = Tensor.softmax x ~dim:0 in
  let log_p = Tensor.log p in
  Tensor.(neg (sum (mul p log_p))) |> Tensor.to_float0_exn

let conditional_entropy x y =
  let joint = Tensor.cat [x; y] ~dim:0 in
  let joint_entropy = shannon_entropy joint in
  let y_entropy = shannon_entropy y in
  joint_entropy -. y_entropy

let joint_probability x y =
  let n = Tensor.shape x |> List.hd in
  let joint = Tensor.stack [x; y] ~dim:1 in
  let unique, counts = Tensor.unique_consecutive joint ~dim:0 ~return_counts:true in
  Tensor.div (Tensor.to_kind Float counts) (Tensor.scalar (float_of_int n))

let conditional_probability x y =
  let joint_prob = joint_probability x y in
  let marginal_prob_x = Tensor.sum joint_prob ~dim:1 ~keepdim:true in
  Tensor.div joint_prob marginal_prob_x

let normal_cdf x =
  0.5 *. (1. +. erf (x /. sqrt 2.))

let jarque_bera_test x =
  let n = Tensor.shape x |> List.hd |> float_of_int in
  let mean = Tensor.mean x |> Tensor.to_float0_exn in
  let std = Tensor.std x ~unbiased:true |> Tensor.to_float0_exn in
  let skewness = Tensor.(mean (pow (div (sub x (scalar mean)) (scalar std)) (scalar 3.))) |> Tensor.to_float0_exn in
  let kurtosis = Tensor.(mean (pow (div (sub x (scalar mean)) (scalar std)) (scalar 4.))) |> Tensor.to_float0_exn in
  let jb = (n /. 6.) *. (skewness ** 2. +. 0.25 *. (kurtosis -. 3.) ** 2.) in
  let p_value = 1. -. chi2_cdf jb 2 in
  (jb, p_value)

let ljung_box_test x lag =
  let n = Tensor.shape x |> List.hd |> float_of_int in
  let acf = Tensor.slice (acf x lag) ~dim:0 ~start:1 ~end:None () in
  let q = Tensor.(sum (div (pow acf (scalar 2.)) 
                          (scalar n -. Tensor.arange ~start:1. ~end_:(float_of_int (lag+1)) ~options:(Kind K.Float, Device.Cpu))))
          |> Tensor.to_float0_exn in
  let q = n *. (n +. 2.) *. q in
  let p_value = 1. -. chi2_cdf q lag in
  (q, p_value)

let kolmogorov_smirnov_test x cdf =
  let n = Tensor.shape x |> List.hd |> float_of_int in
  let sorted_x = Tensor.sort x ~dim:0 ~descending:false in
  let ecdf = Tensor.arange ~start:1. ~end_:(n+.1.) ~options:(Kind K.Float, Device.Cpu) |> Tensor.div_scalar n in
  let tcdf = Tensor.map (fun xi -> cdf (Tensor.to_float0_exn xi)) sorted_x in
  let d = Tensor.max (Tensor.abs (Tensor.sub ecdf tcdf)) |> Tensor.to_float0_exn in
  let p_value = ks_test_p_value d n in
  (d, p_value)

let akaike_information_criterion log_likelihood k =
  2. *. float_of_int k -. 2. *. log_likelihood

let bayesian_information_criterion log_likelihood k n =
  float_of_int k *. log (float_of_int n) -. 2. *. log_likelihood

let quantile tensor q =
  let sorted = Tensor.sort tensor ~dim:0 ~descending:false in
  let index = int_of_float (float_of_int (Tensor.shape sorted |> List.hd) *. q) in
  Tensor.get sorted index |> Tensor.to_float0_exn

let chi2_cdf x df =
  let rec lower_gamma s x =
    if x < s +. 1. then
      let rec series_sum term sum k =
        if term < 1e-10 *. sum then sum
        else
          let new_term = term *. x /. (s +. float_of_int k) in
          series_sum new_term (sum +. new_term) (k + 1)
      in
      exp (s *. log x -. x) *. (series_sum 1. 1. 1) /. exp (log_gamma s)
    else
      1. -. upper_gamma s x
  and upper_gamma s x =
    let rec cf a b =
      let rec loop i a b =
        if i > 100 then a /. b
        else
          let a' = x *. b +. float_of_int i *. a in
          let b' = x *. a' +. (s +. float_of_int i) *. b in
          loop (i + 1) a' b'
      in
      loop 1 1. ((s -. 1.) *. log x +. s *. log 2. +. log_gamma s |> exp)
    in
    exp ((s -. 1.) *. log x -. x) *. cf 1. 1.
  in
  lower_gamma (float_of_int df /. 2.) (x /. 2.)

let ks_test_p_value d n =
  let lambda = (sqrt n +. 0.12 +. 0.11 /. sqrt n) *. d in
  let rec series_sum k acc =
    if k > 100 then acc
    else
      let term = (-1.) ** float_of_int k *. exp (-2. *. lambda *. lambda *. float_of_int k *. float_of_int k) in
      series_sum (k + 1) (acc +. term)
  in
  2. *. series_sum 1 0.

let ljung_box_test x lag =
  let n = Tensor.shape x |> List.hd |> float_of_int in
  let acf = Tensor.slice (acf x lag) ~dim:0 ~start:1 ~end:None () in
  let q = Tensor.(sum (div (pow acf (scalar 2.)) 
                          (scalar n -. Tensor.arange ~start:1. ~end_:(float_of_int (lag+1)) ~options:(Kind K.Float, Device.Cpu))))
          |> Tensor.to_float0_exn in
  let q = n *. (n +. 2.) *. q in
  let p_value = 1. -. chi2_cdf q lag in
  (q, p_value)

let kolmogorov_smirnov_test x cdf =
  let n = Tensor.shape x |> List.hd |> float_of_int in
  let sorted_x = Tensor.sort x ~dim:0 ~descending:false in
  let ecdf = Tensor.arange ~start:1. ~end_:(n+.1.) ~options:(Kind K.Float, Device.Cpu) |> Tensor.div_scalar n in
  let tcdf = Tensor.map (fun xi -> cdf (Tensor.to_float0_exn xi)) sorted_x in
  let d = Tensor.max (Tensor.abs (Tensor.sub ecdf tcdf)) |> Tensor.to_float0_exn in
  let p_value = ks_test_p_value d n in
  (d, p_value)

let akaike_information_criterion log_likelihood k =
  2. *. float_of_int k -. 2. *. log_likelihood

let bayesian_information_criterion log_likelihood k n =
  float_of_int k *. log (float_of_int n) -. 2. *. log_likelihood

let quantile tensor q =
  let sorted = Tensor.sort tensor ~dim:0 ~descending:false in
  let index = int_of_float (float_of_int (Tensor.shape sorted |> List.hd) *. q) in
  Tensor.get sorted index |> Tensor.to_float0_exn

let chi2_cdf x df =
  let rec lower_gamma s x =
    if x < s +. 1. then
      let rec series_sum term sum k =
        if term < 1e-10 *. sum then sum
        else
          let new_term = term *. x /. (s +. float_of_int k) in
          series_sum new_term (sum +. new_term) (k + 1)
      in
      exp (s *. log x -. x) *. (series_sum 1. 1. 1) /. exp (log_gamma s)
    else
      1. -. upper_gamma s x
  and upper_gamma s x =
    let rec cf a b =
      let rec loop i a b =
        if i > 100 then a /. b
        else
          let a' = x *. b +. float_of_int i *. a in
          let b' = x *. a' +. (s +. float_of_int i) *. b in
          loop (i + 1) a' b'
      in
      loop 1 1. ((s -. 1.) *. log x +. s *. log 2. +. log_gamma s |> exp)
    in
    exp ((s -. 1.) *. log x -. x) *. cf 1. 1.
  in
  lower_gamma (float_of_int df /. 2.) (x /. 2.)

let ks_test_p_value d n =
  let lambda = (sqrt n +. 0.12 +. 0.11 /. sqrt n) *. d in
  let rec series_sum k acc =
    if k > 100 then acc
    else
      let term = (-1.) ** float_of_int k *. exp (-2. *. lambda *. lambda *. float_of_int k *. float_of_int k) in
      series_sum (k + 1) (acc +. term)
  in
  2. *. series_sum 1 0.