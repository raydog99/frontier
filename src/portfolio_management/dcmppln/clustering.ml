open Torch
open Types
open Utils

let compute_modularity_matrix (corr_matrix: correlation_matrix) : Tensor.t =
  let n = Tensor.shape corr_matrix |> fst in
  let k = Tensor.sum corr_matrix ~dim:[1] in
  let m = Tensor.sum k |> Tensor.item |> float_of_int in
  let expected = Tensor.div (Tensor.ger k k) (Scalar.float m) in
  Tensor.sub corr_matrix expected

let kmeans_clustering (data: Tensor.t) (k: int) (max_iterations: int) : Tensor.t =
  let n, d = Tensor.shape2_exn data in
  let centroids = Tensor.index_select data 0 (Tensor.randperm k) in
  let rec iterate centroids iter =
    if iter >= max_iterations then centroids
    else
      let distances = Tensor.cdist data centroids in
      let assignments = Tensor.argmin distances ~dim:1 ~keepdim:false in
      let new_centroids = Tensor.zeros [k; d] in
      let counts = Tensor.zeros [k] in
      Tensor.index_add_ new_centroids 0 assignments data;
      Tensor.index_add_ counts 0 assignments (Tensor.ones [n]);
      let new_centroids = Tensor.div new_centroids (Tensor.unsqueeze counts ~dim:1) in
      if Tensor.allclose centroids new_centroids ~rtol:1e-4 ~atol:1e-6 then
        new_centroids
      else
        iterate new_centroids (iter + 1)
  in
  iterate centroids 0

let spectral_clustering (corr_matrix: correlation_matrix) (num_communities: int) : community array =
  log_message "Starting spectral clustering";
  let modularity_matrix = compute_modularity_matrix corr_matrix in
  let eig = eigen_decomposition modularity_matrix |> sort_eigenpairs in
  let top_k_eigenvectors = Tensor.narrow eig.eigenvectors 1 0 num_communities in
  let assignments = kmeans_clustering top_k_eigenvectors num_communities 100 in
  let communities = Array.init num_communities (fun _ -> [||]) in
  for i = 0 to (Tensor.shape assignments |> fst) - 1 do
    let cluster = Tensor.get assignments [i] |> Tensor.item |> int_of_float in
    communities.(cluster) <- Array.append communities.(cluster) [|i|];
  done;
  log_message "Spectral clustering completed";
  communities

let decompose_portfolio (portfolio: portfolio) (communities: community array) : portfolio array =
  Array.map (fun community ->
    let community_assets = Array.map (fun i -> portfolio.assets.(i)) community in
    let community_weights = Tensor.index_select portfolio.weights 0 (float_array_to_tensor (Array.map float_of_int community)) in
    let community_returns = Tensor.index_select portfolio.expected_returns 0 (float_array_to_tensor (Array.map float_of_int community)) in
    { assets = community_assets; weights = community_weights; expected_returns = community_returns }
  ) communities