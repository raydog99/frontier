open Types
open Linalg

type diffusion_kernel = {
  base_kernel: kernel;
  time_steps: float;
  eigenvalues: Tensor.t option;
  eigenfunctions: Tensor.t option;
}

(* Compute diffusion distance between two points *)
let diffusion_distance kernel p x x' =
  let n_samples = Tensor.size x 0 in
  
  (* Compute kernel evaluations *)
  let k_p_x = kernel x (Tensor.unsqueeze x 0) in
  let k_p_x' = kernel x' (Tensor.unsqueeze x' 0) in
  
  (* Compute squared diffusion distance *)
  let diff = Tensor.sub k_p_x k_p_x' in
  let norm = Tensor.norm2 diff in
  Tensor.float_value norm

(* Compute smoothness measure r_K_p *)
let smoothness_measure kernel p f px =
  (* Center the function values *)
  let f_mean = Tensor.mean f in
  let f_centered = Tensor.sub f f_mean in
  
  (* Compute L2 norm in feature space *)
  let norm_px = Tensor.norm2 f_centered in
  let norm_px_sq = Tensor.pow norm_px 2. in
  
  (* Compute RKHS norm using eigendecomposition *)
  let k_matrix = kernel px px in
  let eigenvalues, eigenvectors = eigensystem k_matrix in
  
  (* Transform eigenvalues according to diffusion time *)
  let transformed_eigenvalues = Tensor.pow eigenvalues p in
  
  (* Project centered function onto eigenvectors *)
  let proj = Tensor.matmul (Tensor.transpose eigenvectors 0 1) f_centered in
  let proj_sq = Tensor.pow proj 2. in
  
  (* Compute RKHS norm via spectral formula *)
  let rkhs_norm = Tensor.sum (Tensor.div proj_sq transformed_eigenvalues) in
  
  (* Return ratio *)
  Tensor.div norm_px_sq rkhs_norm

(* Construct diffusion kernel with precomputed eigendecomposition *)
let create_diffusion_kernel base_kernel time_steps x =
  let k_matrix = base_kernel x x in
  let eigenvalues, eigenfunctions = eigensystem k_matrix in
  {
    base_kernel;
    time_steps;
    eigenvalues = Some eigenvalues;
    eigenfunctions = Some eigenfunctions;
  }

(* Apply diffusion kernel *)
let apply_diffusion_kernel dk x y =
  match (dk.eigenvalues, dk.eigenfunctions) with
  | (Some eigenvalues, Some eigenfunctions) ->
      (* Transform eigenvalues according to diffusion time *)
      let transformed_eigenvalues = 
        Tensor.pow eigenvalues dk.time_steps in
      
      (* Project input points *)
      let proj_x = Tensor.matmul x eigenfunctions in
      let proj_y = Tensor.matmul y eigenfunctions in
      
      (* Apply spectral transform *)
      let scaled_proj = 
        Tensor.mul proj_x (Tensor.unsqueeze transformed_eigenvalues 0) in
      
      (* Compute final kernel values *)
      Tensor.matmul scaled_proj (Tensor.transpose proj_y 0 1)
  | _ ->
      (* Fall back to repeated application of base kernel *)
      let rec iterate k_val steps =
        if steps <= 0. then k_val
        else
          let k_next = Tensor.matmul k_val 
            (dk.base_kernel x y) in
          iterate k_next (steps -. 1.)
      in
      iterate (dk.base_kernel x y) dk.time_steps

(* Compute diffusion map embedding *)
let diffusion_map kernel t d x =
  let k_matrix = kernel x x in
  let eigenvalues, eigenvectors = eigensystem k_matrix in
  
  (* Scale eigenvectors by transformed eigenvalues *)
  let transformed_eigenvalues = Tensor.pow eigenvalues t in
  let scaled_eigenvectors = 
    Tensor.mul eigenvectors (Tensor.unsqueeze transformed_eigenvalues 1) in
  
  (* Return top d dimensions *)
  Tensor.narrow scaled_eigenvectors 1 0 d