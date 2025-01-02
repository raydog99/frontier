open Torch

type t =
  | Sample
  | Consistent
  | SSE
  | EBE
  | RTE

let all = [
  ("Sample", Sample);
  ("Consistent", Consistent);
  ("SSE", SSE);
  ("EBE", EBE);
  ("RTE", RTE);
]

let estimate ef estimator =
  let n = ef.EfficientFrontier.n in
  let p = ef.EfficientFrontier.p in
  let sample_mean = ef.EfficientFrontier.mean_vector in
  let sample_cov = ef.EfficientFrontier.covariance_matrix in

  let inv_cov = match estimator with
  | Sample -> Tensor.inverse sample_cov
  | Consistent ->
      let c = float_of_int p /. float_of_int n in
      Tensor.((1.0 - c) * inverse sample_cov)
  | SSE ->
      let scale = float_of_int (n - p - 2) /. float_of_int (n - 1) in
      Tensor.(inverse sample_cov * scale)
  | EBE ->
      let scale1 = float_of_int (n - p - 2) /. float_of_int (n - 1) in
      let scale2 = (float_of_int p ** 2.0 +. float_of_int p -. 2.0) /. (float_of_int (n - 1) *. Tensor.sum sample_cov) in
      Tensor.(scale1 * inverse sample_cov + scale2 * eye p)
  | RTE ->
      let trace = Tensor.trace sample_cov in
      Tensor.(inverse (sample_cov * float_of_int (n - 1) + trace * eye p) * float_of_int p)
  in

  let r_gmv = Tensor.(mm (mm (transpose2 (ones [1; p])) inv_cov) sample_mean
                      / mm (mm (transpose2 (ones [1; p])) inv_cov) (ones [p; 1])) in
  
  let v_gmv = Tensor.(1.0 / (mm (mm (transpose2 (ones [1; p])) inv_cov) (ones [p; 1]))) in
  
  let q = Tensor.(inv_cov - (mm (mm inv_cov (ones [p; 1])) (mm (transpose2 (ones [1; p])) inv_cov))
                            / (mm (mm (transpose2 (ones [1; p])) inv_cov) (ones [p; 1]))) in
  
  let s = Tensor.(mm (mm (transpose2 sample_mean) q) sample_mean) in
  
  (r_gmv, v_gmv, s)