open Torch

let stream_tensor tensor ~f ~batch_size =
  let n = Tensor.size tensor 0 in
  let num_batches = (n + batch_size - 1) / batch_size in
  
  for i = 0 to num_batches - 1 do
    let start_idx = i * batch_size in
    let length = min batch_size (n - start_idx) in
    let batch = Tensor.narrow tensor ~dim:0 ~start:start_idx ~length in
    f batch i
  done

let compute_diffusion_operator features config =
  match config.Config.memory_mode with
  | Standard -> 
      let affinity = Diffusion_geometry.fast_affinity_matrix features 1.0 in
      Diffusion_geometry.build_diffusion_operator affinity
      
  | LowMemory ->
      let n = Tensor.size features 0 in
      let operator = ref (Tensor.zeros [n; n]) in
      
      stream_tensor features ~batch_size:config.batch_size
        ~f:(fun batch batch_idx ->
          stream_tensor features ~batch_size:config.batch_size
            ~f:(fun other_batch other_idx ->
              let distances = Gpu_ops.batch_pairwise_distances batch other_batch config in
              let affinity = Tensor.exp (Tensor.div_scalar 
                (Tensor.neg (Tensor.mul distances distances)) 1.0) in
              
              let start_row = batch_idx * config.batch_size in
              let start_col = other_idx * config.batch_size in
              let _ = Tensor.narrow_copy_ 
                (Tensor.narrow !operator ~dim:0 ~start:start_row 
                   ~length:(Tensor.size batch 0))
                ~dim:1 ~start:start_col ~length:(Tensor.size other_batch 0)
                affinity in
              ()
            )
        );
      
      Diffusion_geometry.build_diffusion_operator !operator
      
  | Distributed ->
      let sparse_chunks = EfficientDiffusion_geometry.build_affinity_matrix_streaming 
        features 1.0 in
      List.hd (EfficientDiffusion_geometry.normalize_affinity_streaming 
        (Array.to_list sparse_chunks))