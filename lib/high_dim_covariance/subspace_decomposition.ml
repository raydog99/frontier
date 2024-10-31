open Torch

type partition = {
  high_eigenspace: Tensor.t;
  medium_eigenspace: Tensor.t;
  low_eigenspace: Tensor.t;
}

let decompose ~matrix ~epsilon =
  let eigenvals, eigenvecs = 
    Numerical_stability.stable_eigendecomposition matrix in
  
  let c1 = 20.0 *. sqrt epsilon in
  let c2 = 20.0 *. epsilon in
  
  (* Partition based on eigenvalue thresholds *)
  let high_mask = Tensor.ge eigenvals (Tensor.full [|1|] c1) in
  let medium_mask = Tensor.logical_and
    (Tensor.lt eigenvals (Tensor.full [|1|] c1))
    (Tensor.ge eigenvals (Tensor.full [|1|] c2)) in
  let low_mask = Tensor.lt eigenvals (Tensor.full [|1|] c2) in
  
  let extract_subspace mask =
    let indices = Tensor.nonzero mask |> Tensor.squeeze ~dim:1 in
    Tensor.index_select eigenvecs ~dim:1 ~index:indices in
  
  {
    high_eigenspace = extract_subspace high_mask;
    medium_eigenspace = extract_subspace medium_mask;
    low_eigenspace = extract_subspace low_mask;
  }

let project_samples ~samples ~subspace =
  Fast_matrix_ops.rect_multiply samples 
    (Tensor.transpose subspace ~dim0:0 ~dim1:1)

let combine_estimates ~high ~medium ~cross_terms ~partition ~epsilon =
  let d = Tensor.size high 0 in
  let result = Tensor.zeros [|d; d|] in
  
  (* High eigenvalue subspace estimate *)
  let high_dim = Tensor.size partition.high_eigenspace 1 in
  let high_proj = Tensor.narrow result ~dim:0 ~start:0 ~length:high_dim
    |> Tensor.narrow ~dim:1 ~start:0 ~length:high_dim in
  Tensor.copy_ high_proj high;
  
  (* Medium eigenvalue subspace estimate *)
  let medium_dim = Tensor.size partition.medium_eigenspace 1 in
  let medium_proj = Tensor.narrow result 
    ~dim:0 ~start:high_dim ~length:medium_dim
    |> Tensor.narrow ~dim:1 ~start:high_dim ~length:medium_dim in
  Tensor.copy_ medium_proj medium;
  
  (* Add cross terms *)
  let cross_terms_scaled = Tensor.mul_scalar cross_terms 
    (1.0 /. sqrt epsilon) in
  add_cross_terms result cross_terms_scaled partition;
  
  (* Ensure symmetry and PSD *)
  Numerical_stability.project_to_psd result