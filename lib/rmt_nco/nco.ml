open Torch

let compute_dissimilarity_matrix correlation =
  Tensor.sqrt (Tensor.mul_scalar (Tensor.sub (Tensor.ones (Tensor.size correlation)) correlation) 0.5)

let minimum_spanning_tree dissimilarity =
  let n = Tensor.size dissimilarity 0 in
  let edges = Tensor.triu_indices n n 1 in
  let weights = Tensor.index dissimilarity edges in
  let sorted_indices = Tensor.argsort weights in
  let sorted_edges = Tensor.index_select edges 1 sorted_indices in
  
  let parent = Array.init n (fun i -> i) in
  let rec find x =
    if parent.(x) != x then parent.(x) <- find parent.(x);
    parent.(x)
  in
  let union x y =
    let px, py = find x, find y in
    if px != py then parent.(px) <- py
  in
  
  let mst_edges = ref [] in
  for i = 0 to Tensor.size sorted_edges 1 - 1 do
    let u, v = Tensor.get sorted_edges 0 i |> Tensor.item_int, Tensor.get sorted_edges 1 i |> Tensor.item_int in
    if find u != find v then begin
      union u v;
      mst_edges := (u, v) :: !mst_edges
    end
  done;
  !mst_edges

let spectral_clustering adjacency k =
  let n = Tensor.size adjacency 0 in
  let degree = Tensor.sum adjacency ~dim:[1] ~keepdim:true in
  let laplacian = Tensor.sub (Tensor.diag degree) adjacency in
  let normalized_laplacian = 
    Tensor.matmul 
      (Tensor.matmul (Tensor.pow degree (-0.5)) laplacian)
      (Tensor.pow degree (-0.5))
  in
  let eigenvalues, eigenvectors = Tensor.symeig ~eigenvectors:true normalized_laplacian in
  let _, top_k_indices = Tensor.topk eigenvalues k in
  let embedding = Tensor.index_select eigenvectors 1 top_k_indices in
  
  (* K-means clustering on the embedding *)
  let centroids = Tensor.rand [k; Tensor.size embedding 1] in
  let max_iter = 100 in
  let rec kmeans iter prev_centroids =
    if iter >= max_iter then prev_centroids
    else
      let distances = Tensor.(
        sum ~dim:[2] ~keepdim:true
          (pow (sub (unsqueeze embedding ~dim:1) (unsqueeze centroids ~dim:0)) (Scalar.float 2.))
      ) in
      let assignments = Tensor.argmin distances ~dim:[1] ~keepdim:false in
      let new_centroids = Tensor.zeros_like centroids in
      for i = 0 to k - 1 do
        let cluster_points = Tensor.masked_select embedding (Tensor.eq assignments (Tensor.full_like assignments i)) in
        let cluster_mean = Tensor.mean cluster_points ~dim:[0] ~keepdim:false in
        Tensor.copy_ (Tensor.narrow new_centroids 0 i 1) cluster_mean
      done;
      if Tensor.allclose new_centroids prev_centroids then new_centroids
      else kmeans (iter + 1) new_centroids
  in
  let final_centroids = kmeans 0 centroids in
  let distances = Tensor.(
    sum ~dim:[2] ~keepdim:true
      (pow (sub (unsqueeze embedding ~dim:1) (unsqueeze final_centroids ~dim:0)) (Scalar.float 2.))
  ) in
  Tensor.argmin distances ~dim:[1] ~keepdim:false

let optimize data =
  let correlation = Tensor.corrcoef data in
  let dissimilarity = compute_dissimilarity_matrix correlation in
  let mst_edges = minimum_spanning_tree dissimilarity in
  
  (* Convert MST edges to adjacency matrix *)
  let n = Tensor.size correlation 0 in
  let adjacency = Tensor.zeros [n; n] in
  List.iter (fun (u, v) ->
    Tensor.set adjacency [u; v] (Tensor.get dissimilarity [u; v]);
    Tensor.set adjacency [v; u] (Tensor.get dissimilarity [v; u])
  ) mst_edges;
  
  (* Estimate optimal number of clusters *)
  let k = max 2 (min 10 (n / 10)) in
  
  let cluster_assignments = spectral_clustering adjacency k in
  
  (* Compute intra-cluster weights *)
  let intra_cluster_weights = Tensor.zeros [n; 1] in
  for i = 0 to k - 1 do
    let cluster_mask = Tensor.eq cluster_assignments (Tensor.full_like cluster_assignments i) in
    let cluster_data = Tensor.masked_select data cluster_mask in
    let cluster_cov = Rmt.estimate_covariance cluster_data `Linear_Shrinkage in
    let cluster_returns = Tensor.mean cluster_data ~dim:[0] ~keepdim:true in
    let cluster_weights = Portfolio.markowitz_optimize cluster_returns cluster_cov 0.1 in (* 10% target return *)
    Tensor.masked_scatter_ intra_cluster_weights cluster_mask cluster_weights
  done;
  
  let cluster_returns = Tensor.zeros [k; Tensor.size data 1] in
  for i = 0 to k - 1 do
    let cluster_mask = Tensor.eq cluster_assignments (Tensor.full_like cluster_assignments i) in
    let cluster_data = Tensor.masked_select data cluster_mask in
    Tensor.copy_ (Tensor.narrow cluster_returns 0 i 1) (Tensor.mean cluster_data ~dim:[0] ~keepdim:true)
  done;
  let cluster_cov = Rmt.estimate_covariance cluster_returns `Linear_Shrinkage in
  let inter_cluster_weights = Portfolio.markowitz_optimize cluster_returns cluster_cov 0.1 in (* 10% target return *)
  
  Tensor.mul intra_cluster_weights (Tensor.index_select inter_cluster_weights 1 cluster_assignments)