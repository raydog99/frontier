open Torch

type batch_config = {
  min_batch_size: int;
  max_batch_size: int;
  target_memory_gb: float;
  adaptation_rate: float;
}

let optimal_batch_size dimension config =
  let bytes_per_float = 4 in
  let memory_per_sample = dimension * dimension * bytes_per_float in
  let target_memory_bytes = 
    config.target_memory_gb *. 1024. *. 1024. *. 1024. in
  let optimal_size = 
    int_of_float (target_memory_bytes /. float_of_int memory_per_sample) in
  min (max optimal_size config.min_batch_size) config.max_batch_size

let batch_mm matrices batch_size =
  let total = List.length matrices in
  let rec process start acc =
    if start >= total then 
      Tensor.cat (List.rev acc) ~dim:0
    else
      let end_idx = min (start + batch_size) total in
      let batch = List.init (end_idx - start) 
        (fun i -> List.nth matrices (start + i)) in
      let batch_tensor = Tensor.stack batch ~dim:0 in
      let result = Tensor.bmm batch_tensor 
        (Tensor.transpose batch_tensor (-1) (-2)) in
      process end_idx (result :: acc)
  in
  process 0 []

let batch_covariance samples batch_size =
  let n = Tensor.size samples 0 in
  let d = Tensor.size samples 1 in
  let mean = Tensor.mean samples ~dim:[0] ~keepdim:true in
  let centered = Tensor.sub samples (Tensor.expand_as mean samples) in
  
  let rec process start acc =
    if start >= n then
      let sum = List.fold_left Tensor.add_ (Tensor.zeros [d; d]) acc in
      Tensor.div_scalar sum (float_of_int n)
    else
      let end_idx = min (start + batch_size) n in
      let batch = Tensor.narrow centered 0 start (end_idx - start) in
      let cov_batch = Tensor.mm 
        (Tensor.transpose batch 0 1) 
        batch in
      process end_idx (cov_batch :: acc)
  in
  process 0 []

let streaming_mean data_generator dimension =
  let mean = Tensor.zeros [dimension] in
  let count = ref 0 in
  
  let rec update () =
    match data_generator () with
    | None -> mean
    | Some batch ->
        let n = Tensor.size batch 0 in
        let weight = float_of_int !count /. 
          float_of_int (!count + n) in
        let batch_mean = 
          Tensor.mean batch ~dim:[0] ~keepdim:false in
        Tensor.mul_scalar_ mean weight;
        Tensor.add_ mean 
          (Tensor.mul_scalar batch_mean 
            (1. -. weight));
        count := !count + n;
        update ()
  in
  update ()

let streaming_covariance data_generator dimension =
  let mean = Tensor.zeros [dimension] in
  let cov = Tensor.zeros [dimension; dimension] in
  let count = ref 0 in
  
  let rec update () =
    match data_generator () with
    | None -> mean, cov
    | Some batch ->
        let n = Tensor.size batch 0 in
        let old_count = float_of_int !count in
        let new_count = float_of_int (!count + n) in
        
        (* Update mean *)
        let batch_mean = 
          Tensor.mean batch ~dim:[0] ~keepdim:false in
        let delta = Tensor.sub batch_mean mean in
        Tensor.mul_scalar_ mean (old_count /. new_count);
        Tensor.add_ mean 
          (Tensor.mul_scalar batch_mean 
            (float_of_int n /. new_count));
        
        (* Update covariance *)
        let centered = 
          Tensor.sub batch 
            (Tensor.expand_as mean batch) in
        let batch_cov = 
          Tensor.mm 
            (Tensor.transpose centered 0 1) 
            centered in
        Tensor.mul_scalar_ cov (old_count /. new_count);
        Tensor.add_ cov 
          (Tensor.mul_scalar batch_cov 
            (float_of_int n /. new_count));
        
        count := !count + n;
        update ()
  in
  update ()