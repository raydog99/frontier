open Torch

module Statistics = struct
  let mean lst =
    let n = float_of_int (List.length lst) in
    List.fold_left (+.) 0.0 lst /. n

  let std lst =
    let mu = mean lst in
    let n = float_of_int (List.length lst) in
    let variance = List.fold_left (fun acc x ->
      acc +. (x -. mu) *. (x -. mu)
    ) 0.0 lst /. (n -. 1.0) in
    sqrt variance

  let median lst =
    let sorted = List.sort compare lst in
    let n = List.length sorted in
    if n mod 2 = 0 then
      let mid = n / 2 in
      (List.nth sorted (mid - 1) +. List.nth sorted mid) /. 2.0
    else
      List.nth sorted (n / 2)

  let quantile lst p =
    let sorted = List.sort compare lst in
    let n = float_of_int (List.length sorted) in
    let pos = p *. (n -. 1.0) in
    let low_idx = int_of_float (floor pos) in
    let high_idx = int_of_float (ceil pos) in
    let weight = pos -. float_of_int low_idx in
    
    let low_val = List.nth sorted low_idx in
    let high_val = List.nth sorted high_idx in
    low_val *. (1.0 -. weight) +. high_val *. weight

  let compute_correlation x y =
    let n = float_of_int (Tensor.shape x).(0) in
    let mean_x = Tensor.mean x |> Tensor.item in
    let mean_y = Tensor.mean y |> Tensor.item in
    
    let x_centered = Tensor.sub x (Tensor.float_scalar mean_x) in
    let y_centered = Tensor.sub y (Tensor.float_scalar mean_y) in
    
    let numerator = Tensor.dot x_centered y_centered |> Tensor.item in
    let denom_x = Tensor.dot x_centered x_centered |> Tensor.item |> sqrt in
    let denom_y = Tensor.dot y_centered y_centered |> Tensor.item |> sqrt in
    
    numerator /. (denom_x *. denom_y)

  let compute_p_value_normal z =
    let u = abs_float z /. sqrt 2.0 in
    let p = 0.5 *. erfc u in
    2.0 *. p  (* Two-tailed test *)

  let compute_p_value_t t df =
    (* Student's t distribution approximation *)
    let x = float_of_int df /. (float_of_int df +. t *. t) in
    let beta = 0.5 *. (float_of_int df) in
    let result = 0.5 *. incomplete_beta beta 0.5 x in
    2.0 *. result  (* Two-tailed test *)

  let compute_p_value_chisq chi_sq df =
    (* Chi-square distribution using gamma function approximation *)
    let k = float_of_int df /. 2.0 in
    let x = chi_sq /. 2.0 in
    1.0 -. incomplete_gamma k x
end