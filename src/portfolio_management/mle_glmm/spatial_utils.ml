open Types
open Torch
open MatrixOps

let distance_matrix locations =
  let n = Tensor.size locations 0 in
  let diff_x = Tensor.sub (Tensor.unsqueeze locations ~dim:1)
                         (Tensor.unsqueeze locations ~dim:0) in
  Tensor.norm diff_x ~p:(Scalar.float 2.0) ~dim:2

let great_circle_distance locations =
  let r = 6371.0 in
  let lat1 = Tensor.select locations ~dim:1 ~index:0 in
  let lon1 = Tensor.select locations ~dim:1 ~index:1 in
  let lat2 = Tensor.unsqueeze lat1 ~dim:1 in
  let lon2 = Tensor.unsqueeze lon1 ~dim:1 in
  
  let dlat = Tensor.sub lat2 (Tensor.transpose lat1 ~dim0:0 ~dim1:1) in
  let dlon = Tensor.sub lon2 (Tensor.transpose lon1 ~dim0:0 ~dim1:1) in
  
  let a = Tensor.(add
    (mul (sin (div dlat (scalar_float 2.0))) 
         (sin (div dlat (scalar_float 2.0))))
    (mul (cos lat1) 
         (mul (cos (transpose lat2 ~dim0:0 ~dim1:1))
              (mul (sin (div dlon (scalar_float 2.0)))
                   (sin (div dlon (scalar_float 2.0))))))) in
  
  let c = Tensor.(mul_scalar (atan2 (sqrt a) (sqrt (sub (scalar_float 1.0) a)))
                            (scalar_float (2.0 *. r))) in
  c

let variogram locations values =
  let dist_mat = distance_matrix locations in
  let n = Tensor.size locations 0 in
  
  (* Compute squared differences *)
  let val_diff = Tensor.sub (Tensor.unsqueeze values ~dim:1)
                           (Tensor.unsqueeze values ~dim:0) in
  let sq_diff = Tensor.pow val_diff (Tensor.scalar_float 2.0) in
  
  (* Bin distances *)
  let max_dist = Tensor.max dist_mat |> Tensor.scalar_to_float |> Scalar.to_float in
  let n_bins = 15 in
  let bin_width = max_dist /. float n_bins in
  
  let bins = Array.make n_bins (0.0, 0.0) in
  
  for i = 0 to n - 1 do
    for j = i + 1 to n - 1 do
      let dist = Tensor.get dist_mat i j |> Scalar.to_float in
      let diff = Tensor.get sq_diff i j |> Scalar.to_float in
      let bin = int_of_float (dist /. bin_width) in
      if bin < n_bins then
        let (sum, count) = bins.(bin) in
        bins.(bin) <- (sum +. diff, count +. 1.0)
    done
  done;
  
  let distances = Tensor.init [n_bins] (fun i -> 
    Scalar.float ((float i +. 0.5) *. bin_width)) in
  let variogram = Tensor.init [n_bins] (fun i ->
    let (sum, count) = bins.(i) in
    if count > 0.0 then Scalar.float (sum /. (2.0 *. count))
    else Scalar.float 0.0) in
  
  (distances, variogram)

let local_indicators locations values =
  let dist_mat = distance_matrix locations in
  let n = Tensor.size locations 0 in
  
  (* Compute spatial weights (inverse distance) *)
  let weights = Tensor.reciprocal dist_mat in
  let diag_mask = Tensor.eye n |> Tensor.bool_tensor in
  let weights = Tensor.where_ weights 
                           ~condition:diag_mask
                           ~other:(Tensor.zeros_like weights) in
  
  (* Row-standardize weights *)
  let row_sums = Tensor.sum weights ~dim:[1] |> Tensor.unsqueeze ~dim:1 in
  let weights = Tensor.div weights row_sums in
  
  (* Compute local Moran's I *)
  let z_scores = Tensor.sub values (Tensor.mean values) in
  let spatial_lag = Tensor.mm weights (Tensor.unsqueeze z_scores ~dim:1) in
  
  Tensor.mul z_scores (Tensor.squeeze spatial_lag ~dim:1)

let kriging_weights spec locations values =
  let dist_mat = distance_matrix locations in
  let cov_mat = Covariance.compute_covariance spec dist_mat in
  
  (* Add nugget effect to diagonal *)
  let n = Tensor.size locations 0 in
  let nugget = Tensor.eye n |> 
               Tensor.mul_scalar (Scalar.float 1e-6) in
  let cov_mat = Tensor.add cov_mat nugget in
  
  (* Solve kriging system *)
  let weights = safe_solve cov_mat values in
  weights