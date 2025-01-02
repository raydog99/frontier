open Torch

let create_twd ?(epsilon=1.0) ?(max_scale=3) ?(use_sliced=false) 
               ?(memory_efficient=false) features =
  (* Clear cache *)
  Cache.clear ();
  
  (* Build diffusion operator *)
  let affinity = Diffusion_geometry.fast_affinity_matrix features epsilon in
  let normalized = Diffusion_geometry.normalize_affinity affinity in
  let operator = Diffusion_geometry.build_diffusion_operator normalized in
  
  (* Embed features *)
  let embedded = Twd_impl.embed_features 
    (Array.init (Tensor.size features 0) (fun i -> Tensor.select features 0 i))
    operator max_scale in
  
  (* Construct tree *)
  let tree = Twd_impl.construct_binary_tree embedded in
  
  (* Return closure over tree for computing distances *)
  fun s1 s2 -> Twd_impl.compute_twd ~use_sliced s1 s2 tree

let create_twd_adaptive ?(epsilon=1.0) ?(max_scale=3) ?(use_sliced=false) 
                       ?(memory_efficient=false) features =
  (* Determine optimal scales *)
  let scales = Adaptive_scaling.compute_optimal_scales features max_scale in
  
  (* Use memory-efficient implementation for large datasets *)
  if memory_efficient then
    let sparse_chunks = EfficientDiffusion_geometry.build_affinity_matrix_streaming 
      features epsilon in
    let normalized_chunks = EfficientDiffusion_geometry.normalize_affinity_streaming 
      sparse_chunks in
    
    (* Process each scale *)
    let scale_embeddings = List.map (fun scale ->
      Twd_impl.compute_scale_embedding_sparse 
        (Array.init (Tensor.size features 0) (fun i -> Tensor.select features 0 i))
        (List.hd normalized_chunks)
        (int_of_float scale)
    ) scales in
    
    (* Construct tree using scale embeddings *)
    let tree = Twd_impl.construct_binary_tree_parallel 
      (Array.of_list scale_embeddings) in
    
    fun s1 s2 -> Twd_impl.compute_twd ~use_sliced s1 s2 tree
    
  else
    (* Use standard implementation *)
    create_twd ~epsilon ~max_scale ~use_sliced features

let create_twd_gpu ?(config=Config.default) features =
  Cache.clear ();
  
  let operator = MemoryEfficientOps.compute_diffusion_operator features config in
  
  (* Determine optimal scales *)
  let scales = Adaptive_scaling.compute_optimal_scales features 3 in
  
  (* Compute embeddings *)
  let n = Tensor.size features 0 in
  let embedded = Array.init n (fun i ->
    let feature = Tensor.select features 0 i in
    List.map (fun scale ->
      Twd_impl.compute_scale_embedding_sparse [|feature|] operator (int_of_float scale)
    ) scales |> Array.of_list
  ) in
  
  (* Construct tree *)
  let tree = Twd_impl.construct_binary_tree_parallel embedded in
  
  (* Return GPU-accelerated distance computation function *)
  fun s1 s2 -> Twd_impl.compute_twd_gpu s1 s2 tree config