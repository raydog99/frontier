open Torch

module Manifold = struct
  type t = {
    dim: int;                   
    ambient_dim: int;           
    points: Tensor.t;           
  }

  let create points dim ambient_dim = {
    dim;
    ambient_dim;
    points
  }

  let pairwise_distances points =
    let expanded_a = Tensor.unsqueeze points ~dim:1 in
    let expanded_b = Tensor.unsqueeze points ~dim:0 in
    let diff = Tensor.(expanded_a - expanded_b) in
    Tensor.(sum (diff * diff) ~dim:[2])

  let geodesic_distances points k =
    let dist_matrix = pairwise_distances points in
    let n = Tensor.size dist_matrix |> List.hd in
    let dist = Tensor.clone dist_matrix in
    (* Floyd-Warshall algorithm *)
    for k = 0 to n - 1 do
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let through_k = Tensor.float_value Tensor.(get dist [i; k] + get dist [k; j]) in
          let direct = Tensor.float_value Tensor.(get dist [i; j]) in
          if through_k < direct then
            Tensor.set dist [i; j] through_k
        done
      done
    done;
    dist
end

module LaplaceBeltrami = struct
  type eigensystem = {
    eigenvalues: Tensor.t;
    eigenfunctions: Tensor.t;
  }

  let weight_matrix points epsilon =
    let dist_sq = Manifold.pairwise_distances points in
    Tensor.(exp (dist_sq * (-1. /. (4. *. epsilon))))

  let normalized_laplacian points epsilon =
    let w = weight_matrix points epsilon in
    let d = Tensor.sum1 ~dim:[1] w ~keepdim:true in
    let d_inv_sqrt = Tensor.(pow d (-0.5)) in
    let n = Tensor.size points |> List.hd in
    let l = Tensor.(eye n) in
    Tensor.(l - (d_inv_sqrt * w * d_inv_sqrt))

  let compute_eigensystem laplacian k =
    let eigenvalues, eigenvectors = Tensor.linalg_eigh laplacian in
    let k = min k (List.hd (Tensor.size eigenvalues)) in
    let eigenvalues = Tensor.narrow eigenvalues ~dim:0 ~start:0 ~length:k in 
    let eigenvectors = Tensor.narrow eigenvectors ~dim:1 ~start:0 ~length:k in
    {eigenvalues; eigenfunctions = eigenvectors}

  let estimate_bounds k dim =
    let c_low = 4. *. Float.pi *. Float.pi /. 
      (Float.pow (float_of_int dim) (2. /. float_of_int dim)) in
    let c_up = 4. *. Float.pi *. Float.pi *. 
      float_of_int dim *. float_of_int dim in
    let k_float = float_of_int k in
    (c_low *. Float.pow k_float (2. /. float_of_int dim),
     c_up *. Float.pow k_float (2. /. float_of_int dim))
end

module HeatKernel = struct
  type t = {
    epsilon: float;
    truncation: int;
    eigenvalues: Tensor.t;
    eigenvectors: Tensor.t;
  }

  let create epsilon k eigensystem = {
    epsilon;
    truncation = k;
    eigenvalues = eigensystem.LaplaceBeltrami.eigenvalues;
    eigenvectors = eigensystem.LaplaceBeltrami.eigenfunctions;
  }

  let evaluate kernel x y t =
    let diff = Tensor.(x - y) in
    let dist_sq = Tensor.(sum (diff * diff)) |> Tensor.float_value in
    let d = Tensor.size x |> List.hd |> float_of_int in
    let scale = 1. /. Float.pow (4. *. Float.pi *. t) (d /. 2.) in
    scale *. exp (-. dist_sq /. (4. *. t))

  let build_matrix kernel points t =
    let exp_eigenvalues = Tensor.(exp (kernel.eigenvalues * (-. t))) in
    let scaled_eigenvectors = 
      Tensor.(kernel.eigenvectors * exp_eigenvalues) in
    Tensor.matmul kernel.eigenvectors 
      Tensor.(transpose scaled_eigenvectors ~dim0:0 ~dim1:1)

  let local_approximation x y t =
    let diff = Tensor.(x - y) in
    let dist_sq = Tensor.(sum (diff * diff)) |> Tensor.float_value in
    let d = Tensor.size x |> List.hd |> float_of_int in
    (1. /. Float.pow (4. *. Float.pi *. t) (d /. 2.)) *. 
    exp (-. dist_sq /. (4. *. t))
end