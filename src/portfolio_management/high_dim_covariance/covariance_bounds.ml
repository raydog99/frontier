open Torch

let compute_initial_bound ~samples ~epsilon =
  let n = Tensor.size samples 0 in
  let subset_size = Int.of_float ((1.0 -. epsilon) *. Float.of_int n) in
  
  (* Find samples with smallest norm *)
  let norms = Tensor.stack 
    (Array.init n (fun i ->
      let sample = Tensor.slice samples ~dim:0 ~start:i ~end_:(i+1) in
      Tensor.norm sample)) ~dim:0 in
  let _, indices = Tensor.sort norms in
  let subset_indices = Tensor.slice indices ~dim:0 ~start:0 
    ~end_:subset_size in
  
  (* Compute initial estimate from subset *)
  let subset = Tensor.index_select samples ~dim:0 ~index:subset_indices in
  let cov = Tensor.mm 
    (Tensor.transpose subset ~dim0:0 ~dim1:1) subset in
  Tensor.div_scalar cov (Float.of_int subset_size)
  |> Tensor.mul_scalar 2.0

let verify_bound ~bound ~true_cov =
  (* Check if bound ≥ true_cov *)
  let diff = Tensor.sub bound true_cov in
  let eigenvals = Tensor.eigenvalues diff in
  let min_eigenval = Tensor.min eigenvals |> Tensor.float_value in
  min_eigenval >= -1e-10

let improve_bound ~current ~mean_est ~samples ~epsilon =
  let d = Tensor.size samples 1 in
  let mean_matrix = Tensor.reshape mean_est [|d; d|] in
  let improved = Tensor.add mean_matrix 
    (Tensor.mul_scalar current (sqrt epsilon)) in
  
  (* Ensure PSD *)
  let eigenvals, eigenvecs = 
    Numerical_stability.stable_eigendecomposition improved in
  let pos_eigenvals = Tensor.relu eigenvals in
  Tensor.mm 
    (Tensor.mm eigenvecs (Tensor.diag pos_eigenvals))
    (Tensor.transpose eigenvecs ~dim0:0 ~dim1:1)