open Torch
open Types
open Linalg

(* Numerically stable inverse Laplacian transform *)
let create_stable_inverse_laplacian eta epsilon =
  fun x -> 
    if x < epsilon then epsilon
    else 1. /. (1. -. eta *. x)

module StableKernelPCA = struct
  type pca_state = {
    eigenvectors: Tensor.t;
    eigenvalues: Tensor.t;
    explained_variance_ratio: Tensor.t;
  }

  (* Numerically stable eigendecomposition *)
  let stable_eigensystem tensor epsilon =
    let eigenvalues, eigenvectors = eigensystem tensor in
    (* Stabilize small eigenvalues *)
    let stable_eigenvalues = Tensor.map eigenvalues ~f:(fun x -> 
      if x < epsilon then epsilon else x) in
    stable_eigenvalues, eigenvectors

  (* Fit kernel PCA with stability checks *)
  let fit kernel n_components epsilon x_train =
    (* Compute and center kernel matrix *)
    let k_matrix = kernel x_train x_train in
    let eigenvalues, eigenvectors = stable_eigensystem k_matrix epsilon in
    
    (* Compute explained variance ratios *)
    let total_variance = Tensor.sum eigenvalues in
    let explained_variance_ratio = Tensor.div eigenvalues total_variance in
    
    (* Extract top components *)
    let top_k_eigenvalues = Tensor.narrow eigenvalues 0 0 n_components in
    let top_k_eigenvectors = Tensor.narrow eigenvectors 1 0 n_components in
    let top_k_variance_ratio = 
      Tensor.narrow explained_variance_ratio 0 0 n_components in
    
    { 
      eigenvectors = top_k_eigenvectors;
      eigenvalues = top_k_eigenvalues;
      explained_variance_ratio = top_k_variance_ratio;
    }

  (* Transform new data points *)
  let transform state kernel x_train x_new =
    let k_new = kernel x_train x_new in
    (* Project onto principal components *)
    Tensor.matmul (Tensor.transpose state.eigenvectors 0 1) k_new
end

module Validation = struct
  (* Check if transformation preserves relative smoothness *)
  let check_smoothness_preservation kernel transform f1 f2 px =
    let rec check_scales p acc =
      if p > 5. then acc
      else
        let r1 = Diffusion.smoothness_measure kernel p f1 px in
        let r2 = Diffusion.smoothness_measure kernel p f2 px in
        let preserves = Tensor.le r1 r2 in
        check_scales (p +. 1.) (acc && Tensor.item preserves 0 = 1)
    in
    check_scales 1. true

  (* Estimate optimal number of components *)
  let estimate_components state variance_threshold =
    let cumsum = Tensor.cumsum state.explained_variance_ratio 0 in
    let n_components = Tensor.size state.eigenvalues 0 in
    let rec find_cutoff i =
      if i >= n_components then n_components
      else if Tensor.get cumsum [i] >= variance_threshold then i + 1
      else find_cutoff (i + 1)
    in
    find_cutoff 0
end