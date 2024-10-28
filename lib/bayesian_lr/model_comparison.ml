open Torch
open Type

let log_likelihood y pred sigma =
  let n = float_of_int (size y 0) in
  let residuals = sub y pred in
  let ss = dot residuals residuals in
  -0.5 *. (n *. log (2. *. Float.pi *. sigma) +. float_of_elt ss /. sigma)

let compute_dic samples x y =
  let n_samples = List.length samples in
  
  (* Mean deviance *)
  let mean_dev = List.fold_left (fun acc s ->
    let pred = matmul x s.theta_star in
    acc +. (-2.) *. log_likelihood y pred s.sigma
  ) 0. samples /. float_of_int n_samples in
  
  (* Point estimate deviance *)
  let mean_theta = List.fold_left (fun acc s -> add acc s.theta_star) 
    (zeros_like (List.hd samples).theta_star) samples in
  let mean_theta = scalar_mul (1. /. float_of_int n_samples) mean_theta in
  let mean_sigma = List.fold_left (fun acc s -> acc +. s.sigma) 0. samples 
    /. float_of_int n_samples in
  
  let point_dev = -2. *. log_likelihood y (matmul x mean_theta) mean_sigma in
  
  let p_eff = mean_dev -. point_dev in
  let dic = point_dev +. 2. *. p_eff in
  
  {dic; waic = 0.; lpd = -.point_dev /. 2.; p_eff}

let compute_waic samples x y =
  let n = size y 0 in
  let n_samples = List.length samples in
  
  (* Compute pointwise log-likelihood *)
  let ll_matrix = zeros [n; n_samples] in
  List.iteri (fun j s ->
    let pred = matmul x s.theta_star in
    for i = 0 to n - 1 do
      let yi = get y i in
      let predi = get pred i in
      let ll = -0.5 *. (log (2. *. Float.pi *. s.sigma) +. 
                       (yi -. predi) ** 2. /. s.sigma) in
      tensor_set ll_matrix [i; j] ll
    done
  ) samples;
  
  (* Compute WAIC components *)
  let lpd = sum (log (mean (exp ll_matrix) ~dim:[1] ~keepdim:true)) |> float_of_elt in
  let p_waic = sum (variance ll_matrix ~dim:[1] ~unbiased:true) |> float_of_elt in
  let waic = -2. *. (lpd -. p_waic) in
  
  {dic = 0.; waic; lpd; p_eff = p_waic}