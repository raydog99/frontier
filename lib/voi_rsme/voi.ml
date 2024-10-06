open Torch

let rmse_lower_bound information sigma_x =
  sigma_x *. exp (-. information)

let calculate_mutual_information predictors response =
  try
    let open Tensor in
    (* Gaussian formula for Shannon's mutual information *)
    let k_z = mm (transpose predictors 0 1) predictors in
    let k_x = mm (transpose response 0 1) response in
    let k_zx = cat [k_z; mm (transpose predictors 0 1) response] ~dim:1 in
    let k_zx = cat [k_zx; cat [mm (transpose response 0 1) predictors; k_x] ~dim:1] ~dim:0 in
    let det_k_z = Linalg.det k_z in
    let det_k_x = Linalg.det k_x in
    let det_k_zx = Linalg.det k_zx in
    0.5 *. (log det_k_z +. log det_k_x -. log det_k_zx)
  with _ ->
    failwith "Failed to calculate mutual information"